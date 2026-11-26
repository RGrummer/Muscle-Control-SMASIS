'''Run rough mujoco model without muscles (direct force application) using level MPC'''
import mujoco
from mujoco import viewer
import time
import numpy as np
import matplotlib.pyplot as plt
import cyipopt
import jax
import jax.numpy as jnp
import jax.random as jr

### Controller for moving end effector (intersection point of the unactuated links) along a path
### State x = [theta_r, theta_l, dtheta_r, dtheta_l]
### Control u = [F_r, F_l] force on right actuated rod, force on left actuated rod

# %% MUJOCO SETUP
# Load model
# model = mujoco.MjModel.from_xml_path("rough_mujoco_model_v2/robot.xml")
# data = mujoco.MjData(model)
# key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "save_1")
# mujoco.mj_resetDataKeyframe(model, data, key_id) # Load keyframe
model = mujoco.MjModel.from_xml_path("proper_mujoco_model/robot.xml")
data = mujoco.MjData(model)

t_final = 15.0 # s, Time length of sim
loops = 2
dt = 0.01 # s, Timestep of sim
model.opt.timestep = dt
Nt = int(t_final/dt) # Number of timesteps

## Get link properties from model
l1 = 0.1530 #np.linalg.norm(model.body('r2').pos)
l2 = 0.1157 #np.linalg.norm(model.site('EE').pos)
m1 = model.body('r1').mass[0]
I1 = model.body('r1').inertia[0]
m2 = model.body('r2').mass[0]
I2 = model.body('r2').inertia[0]

## Morphology variables
delta_x = 1.853e-02
delta_y = 5.375e-03
r_muscle_attach = 9.572e-03
l1 = 1.530e-01
l2 = 1.157e-01
k = 0.01463796416065618 # Nm/rad, spring constant for springs on arms
k_offset = 0.11784743343532078 # rad, angle offset of spring 0 point off of vertical (positive is towards side of respective arm aka more relaxed)

## Calculated morphology variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

def muscle_length_to_angle(muscle_length):
    'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
    theta = np.pi/2 + phi - np.arccos((r_muscle_attach**2 + r_OC**2 - muscle_length**2)/(2*r_muscle_attach*r_OC))
    return theta

def angle_to_muscle_length(theta):
    'Finds muscle length given specific angle of actuated arm'
    # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
    muscle_length = jnp.sqrt(delta_x**2 + delta_y**2 + r_muscle_attach**2
                             + 2*delta_y*r_muscle_attach*jnp.cos(theta)
                             - 2*delta_x*r_muscle_attach*jnp.sin(theta))
    return muscle_length

def muscle_force_to_moment(F, muscle_length, theta = None):
    'Converts force from muscle to moment on the respective arm'
    'MUST provide either muscle_length or theta'
    if theta is None:
        theta = muscle_length_to_angle(muscle_length)
    M = r_muscle_attach*F*(delta_x*np.cos(theta) + delta_y*np.sin(theta))/muscle_length
    return M



# %% MPC SETUP
### Uses moving horizon MPC

## Simulation Variables
Nx = 4 # Number of states
Nu = 2 # Number of controls
Nt_h = 15 # Number of timesteps in horizon
t_final_h = Nt_h*dt # s, Time length of horizon
n_nlp = Nx*Nt_h + Nu*(Nt_h - 1)

## Cost coefficients
pos_coeff = 2e3
vel_coeff = 1e1
control_coeff = 1e-1

## Cost matrices
Q = jnp.diag(jnp.array([pos_coeff]*2 + [vel_coeff]*2)) # State cost
R = jnp.diag(jnp.full((2,), control_coeff)) # Control cost
Qf = Q.copy() # Terminal cost = same as Q

## I3 value estimates based on Sukhnandan et al. (2024)
L0 = 0.016 # m, rest length of I3
Lmto = L0/0.89 # m, length of max I3 force
Fmto = 1.61 # N, max force of I3 at Lmto

## Muscle limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = Lmto*(1+percent_L_range) # Max I3 operation length
L_min = Lmto*(1-percent_L_range) # Min I3 operation length

## Control limits
muscle_force_min = 0



# %% TRAJECTORY

def inverse_kinematics(p):
    'Calculates IK of 5 bar linkage for actuated arm angles giveen end effector position p=[x,y]'
    acos_term = jnp.arccos((p[0]**2 + p[1]**2 + l1**2 - l2**2)/(2*l1*jnp.sqrt(p[0]**2 + p[1]**2)))
    atan_term = jnp.arctan(p[0]/p[1])
    theta_r = jnp.pi/2 - (acos_term + atan_term) # Pi terms appear there because this math was originally done for the "relative" thetas off vertical
    theta_l = (acos_term - atan_term) + jnp.pi/2
    return jnp.array([theta_r, theta_l])

def forward_kinematics(x):
    'Calculate forward kinematics of 5 bar linkage end effector x and y positions'
    mid = jnp.array([l1/2*(jnp.cos(x[0]) + jnp.cos(x[1])), 
                     l1/2*(jnp.sin(x[0]) + jnp.sin(x[1]))]) # Midpoint between actuated arms
    avg_angle = (x[0] + x[1])/2
    h = jnp.sqrt(l2**2 - l1**2/4*(2 - 2*jnp.cos(x[0] - x[1]))) # Equation for dist from line between actuated arm ends to end effector
    e_perp = jnp.array([jnp.cos(avg_angle), jnp.sin(avg_angle)])
    endpoint = mid + h * e_perp # Two possible intersection points, choose higher "up" point
    return endpoint

def circle_reference_trajectory():
    'Circle reference trajectory for end effector'
    center = np.array([0.0, 0.185])  # m
    radius = 0.025 # m
    return np.array([center + radius * jnp.array([jnp.cos(2*jnp.pi*i), jnp.sin(2*jnp.pi*i)])
        for i in np.linspace(0, 1, num=Nt, endpoint=False)])#+1)[:-1]]) # Returns an [Nt, 2] array

def heart_reference_trajectory():
    'Heart shaped reference trajectory for end effector'
    center = np.array([0.0, 0.175])  # m
    scale = 0.0017  # scales to about 0.03 m width/height

    heart_points = np.array([
        center + scale * np.array([
            16 * np.sin(i)**3,
            13 * np.cos(i) - 5 * np.cos(2*i) - 2 * np.cos(3*i) - np.cos(4*i)
        ])
        for i in np.linspace(0, 2*np.pi, num=Nt, endpoint=False)
    ])
    return heart_points # Returns an [Nt, 2] array

def angle_trajectory(end_traj_func):
        'Calculates angle trajectory of a reference end effector trajectory'
        end_traj = end_traj_func()
        angle_traj = jax.vmap(inverse_kinematics)(end_traj)
        return angle_traj # Returns an [Nt, 2] array

def reference_trajectory(end_traj_func):
    'Develop a reference trajectory for the linkage'
    angle_traj = angle_trajectory(end_traj_func)
    
    ## Find reference velocities
    angvel_traj = jnp.zeros_like(angle_traj)
    def angvel_traj_central(i, vel):
        return vel.at[i].set((angle_traj[i+1] - angle_traj[i-1]) / (2 * dt))
    angvel_traj = jax.lax.fori_loop(1, Nt - 1, angvel_traj_central, angvel_traj)
    
    x_traj = jnp.hstack([angle_traj, angvel_traj]) # [Nt, Nx] array
    u_traj = jnp.zeros([Nt, Nu])
    
    return x_traj, u_traj

# Target joint positions (in radians)
trag_gen = circle_reference_trajectory
# trag_gen = heart_reference_trajectory
ee_ref = trag_gen()
xref, uref = reference_trajectory(trag_gen)
xref_arr = jnp.stack(xref, axis=1)   # shape (Nx, Nt)
xref_arr_double = jnp.hstack([xref_arr, xref_arr]) # Used to allow for looping
uref_arr = jnp.stack(uref, axis=1)   # shape (Nu, Nt)



# %% SYSTEM DYNAMICS

## Define kinematics
def theta_2s(x):
    'Compute angles of the unactuated links'
    arg = (l1 / l2) * jnp.sin((x[1] - x[0]) / 2.0)
    eps = 1e-12
    arg_clipped = jnp.clip(arg, -1.0 + eps, 1.0 - eps)
    acos_term = jnp.arccos(arg_clipped)
    # acos_term = jnp.arccos(l1/l2*jnp.sin((x[1]-x[0])/2))
    avg = (x[0] + x[1])/2
    theta_r2 = jnp.pi/2 - acos_term + avg # Angle of right side unactuated link
    theta_l2 = acos_term + avg - jnp.pi/2 # Angle of left side unactuated link
    # jax.debug.print("theta_2s: x0={x0}, x1={x1}, acos={acos}, avg={avg}", x0=x[0], x1=x[1], acos=acos_term, avg=avg)
    return theta_r2, theta_l2

def theta_2s_analytic_jac(x, eps=1e-12):
    s = 0.5*(x[1] - x[0])
    a = l1/l2
    sin_s = jnp.sin(s)
    cos_s = jnp.cos(s)
    arg = a*sin_s

    arg_clipped = jnp.clip(arg, -1.0 + eps, 1.0 - eps) # clamp arg to avoid arccos domain issues
    denom = jnp.sqrt(1.0 - arg_clipped**2) # safe denominator sqrt(1 - arg^2)

    darg=a/2*cos_s

    ## Jacobian entries
    jr0 = -darg/denom + 0.5
    jr1 = darg/denom + 0.5
    jl0 = darg/denom + 0.5
    jl1 = -darg/denom + 0.5

    J = jnp.stack([jnp.stack([jr0, jr1]), jnp.stack([jl0, jl1])])
    # jax.debug.print("arg_clipped={arg}, denom={denom}", arg=arg_clipped, denom=denom)
    return J

def dtheta_2s_analytic(x):
    """Compute angular velocities of unactuated links using analytic jacobian"""
    J = theta_2s_analytic_jac(x)
    dthetas = J @ x[2:]
    return dthetas[0], dthetas[1]

## Use Lagrange mechanics to solve the dynamics
def potential_energy(x):
    'Compute total PE of the system (just springs on the actuated arms)'
    V = 0.5*k*((jnp.pi/2 - x[0] - k_offset)**2 + (x[1] - jnp.pi/2 - k_offset)**2)
    return V

def kinetic_energy(x):
    'Compute total KE of the system'
    theta_r2, theta_l2 = theta_2s(x)
    dtheta_r2, dtheta_l2 = dtheta_2s_analytic(x)
    
    ## Math for COM velocity of unactuated links
    r_r = l1*jnp.array([jnp.cos(x[0]), jnp.sin(x[0])])
    r_l = l1*jnp.array([jnp.cos(x[1]), jnp.sin(x[1])])
    r_r2 = l2*jnp.array([jnp.cos(theta_r2), jnp.sin(theta_r2)])
    r_l2 = l2*jnp.array([jnp.cos(theta_l2), jnp.sin(theta_l2)])
    # v_com_r2 = jnp.linalg.norm(r_r*x[2] + r_r2/2*dtheta_r2)
    v_com_r2 = jnp.sqrt(jnp.sum((r_r*x[2] + r_r2/2*dtheta_r2)**2) + 1e-12)
    # v_com_l2 = jnp.linalg.norm(r_l*x[3] + r_l2/2*dtheta_l2)
    v_com_l2 = jnp.sqrt(jnp.sum((r_l*x[3] + r_l2/2*dtheta_l2)**2) + 1e-12)
    
    T = 0.5*(I1 + m1*l1**2/4)*(x[2]**2 + x[3]**2)   \
        + 0.5*I2*(dtheta_r2**2 + dtheta_l2**2)    \
        + 0.5*m2*(v_com_r2**2 + v_com_l2**2)
    # jax.debug.print("dtheta={dtheta_r2},{dtheta_l2}", dtheta_r2=dtheta_r2, dtheta_l2=dtheta_l2)
    return T

def lagrangian(x):
    return kinetic_energy(x) - potential_energy(x) # L = T - V

def dL_dq(x):
    return jax.grad(lagrangian, argnums=0)(x)[:2] # dL/dtheta_r, dL/dtheta_l

def dL_dqdot(x):
    return jax.grad(lagrangian, argnums=0)(x)[2:] # dL/ddtheta_r, dL/ddtheta_l

## Rearrange d/dt*(dL/dqdot) - dL/dq = Q => M*qddot + C - dL/dq = Q
def psuedo_M_and_C(x):
    jac = jax.jacrev(dL_dqdot)(x)
    return jac[:,2:], jac[:,:2] @ x[2:] # d^2L/dqdot^2, (d^2L/dqdqdot)*dq

def muscle_force_to_moment_given_angle(theta, F):
    'Converts force from muscle to moment on the respective arm'
    # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
    muscle_length = angle_to_muscle_length(theta)
    M = r_muscle_attach*F*(delta_x*jnp.cos(theta) + delta_y*jnp.sin(theta))/muscle_length
    return M

def dynamics(x, u):
    'Calculate derivative of state using lagrange mechanics given specific x and u'
    M, C = psuedo_M_and_C(x)
    eps = 1e-10
    M_reg = M + eps * jnp.eye(M.shape[0])
    dL_dq_val = dL_dq(x)
    moment = jnp.array([-muscle_force_to_moment_given_angle(jnp.pi/2 - x[0], u[0]), # right side moment flips sign for absolute coords
                        muscle_force_to_moment_given_angle(x[1] - jnp.pi/2, u[1])])
    
    ddq = jnp.linalg.solve(M_reg, moment - C + dL_dq_val)
    # jax.debug.print("ddq={ddq}, M={M}, C={C}", ddq=ddq, M=M, C=C)
    dx = jnp.concatenate([x[2:], ddq])
    
    return dx

def rk4_dynamics(x, u, dt):
    'Use Runga-Kutta 4 to approximate state change over one timestep'
    f = lambda s: dynamics(s, u)
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6



# %% COST FUNCTIONS AND INDEXING UTILITIES
## Note that only xref_arr and uref_arr operate in absolute index coordinates for position around the circle

def stage_cost(x, u, i, i_offset):
    i_cur = (i + i_offset) % Nt # Allows for moving horizon
    dx = x - xref_arr[:, i_cur]
    du = u - uref_arr[:, i_cur]
    return 0.5 * dx @ Q @ dx + 0.5 * du @ R @ du

def terminal_cost(x, i_offset):
    i_end = (i_offset + Nt_h - 1) % Nt # Final state in horizon
    dx = x - xref_arr[:, i_end]
    return 0.5 * dx @ Qf @ dx


def cost(ztraj, i_offset): # ztraj is 1D vector of [x0, u0, x1, u1, ... , uN-1, xN]
    x_traj, u_traj = build_traj(ztraj)

    # vectorize stage_cost over time indices 0…Nt-2
    stage_indices = jnp.arange(Nt_h - 1)
    J_stages = jnp.sum(jax.vmap(stage_cost, in_axes=(0,0,0, None))(x_traj[:-1], u_traj, stage_indices, i_offset))
    return J_stages + terminal_cost(x_traj[-1], i_offset)


def unpack(ztraj):
    """
    Return:
      z_mat   -- shape (Nt-1, Nx+Nu), each row = [x_k, u_k]
      x_term  -- shape (Nx,), terminal state
    """
    flat_len = (Nt_h - 1) * (Nx + Nu)
    z_mat = jnp.reshape(ztraj[:flat_len], (Nt_h - 1, Nx + Nu))
    x_term = ztraj[flat_len:]
    return z_mat, x_term

def build_traj(ztraj):
    z_mat, x_term = unpack(ztraj)
    x_traj = jnp.vstack([z_mat[:, :Nx], x_term[jnp.newaxis, :]])  # shape (Nt_h, Nx)
    u_traj = z_mat[:, Nx:]  # shape (Nt_h-1, Nu)
    return x_traj, u_traj

## Compile & auto-differentiate
cost_jit = jax.jit(cost)
cost_gradient_jit = jax.jit(jax.grad(cost, argnums=0))
# cost_jit = jax.jit(cost)
# cost_gradient_jit = jax.jit(jax.jacfwd(cost))
hessian_cost = jax.jit(jax.hessian(cost))



# %% CONSTRAINT FUNCTIONS
### Constraint functions work by you having them output values and later creating upper and lower bound matrices for them

## Set up constraint arrays
init_end = 0 + Nx
dyn_end = init_end + Nx*(Nt_h - 1)
state_end = dyn_end + Nt_h*2 # State constraints on thetas
control_end = state_end + (Nt_h - 1)*2
m_nlp = control_end # number of constraints

c_init_inds    = jnp.arange(0, init_end)
c_dyn_inds     = jnp.arange(init_end, dyn_end)
c_state_inds   = jnp.arange(dyn_end, state_end)
c_control_inds = jnp.arange(state_end, control_end)

## Calculate state limits
theta_max = muscle_length_to_angle(L_min)
theta_min = muscle_length_to_angle(L_max)

## Constrain array lower and upper bounds
cl = np.zeros(m_nlp) # Lower constraint bound
cu = np.zeros(m_nlp) # Upper constraint bound
cl[c_state_inds] = [np.pi/2 - theta_max, np.pi/2 + theta_min]*Nt_h
cu[c_state_inds] = [np.pi/2 - theta_min, np.pi/2 + theta_max]*Nt_h
cl[c_control_inds] = 0.0
cu[c_control_inds] = Fmto

def dynamics_constraint(ztraj):
    xtraj, utraj = build_traj(ztraj)
    
    d_init = jnp.zeros((Nt_h - 1, Nx))
    
    def single_dynamics_inst(s, d):
        x_s = xtraj[s]
        u_s = utraj[s]
        
        fx = rk4_dynamics(x_s, u_s, dt)
        df = fx - xtraj[s + 1]
        return d.at[s].set(df) # this is how we update a number or matrix in a loop in Jax
    
    ## This function encodes a loop in jax
    d = jax.lax.fori_loop(
        0, # starting index
        Nt_h - 1, # ending index
        single_dynamics_inst, # function to loop over. It accepts an index (s) and a vector d that is the same shape as d_init, which is the thing that we modify in the loop
        d_init) # vector with the same shape as what we return from the loop

    # Stuff into constraint vector
    c = jnp.zeros(m_nlp)
    c = c.at[c_dyn_inds].set(jnp.reshape(d, (-1,)))
    return c

def state_constraint(ztraj):
    x_traj, _ = build_traj(ztraj)
    
    thetas = jnp.zeros(Nt_h*2)
    def get_theta(i, theta_vec):
        x = x_traj[i]
        theta_vec = theta_vec.at[2*i].set(x[0])
        theta_vec = theta_vec.at[2*i + 1].set(x[1])
        return theta_vec
    thetas = jax.lax.fori_loop(0, Nt_h, get_theta, thetas)
    
    c = jnp.zeros(m_nlp)
    c = c.at[c_state_inds].set(thetas)
    return c

def control_constraint(ztraj):
    _, u_traj = build_traj(ztraj)
    
    us = jnp.zeros((Nt_h-1)*2)
    def get_u(i, u_vec):
        u = u_traj[i]
        u_vec = u_vec.at[2*i].set(u[0])
        u_vec = u_vec.at[2*i + 1].set(u[1])
        return u_vec
    us = jax.lax.fori_loop(0, Nt_h-1, get_u, us)
    
    c = jnp.zeros(m_nlp)
    c = c.at[c_control_inds].set(us)
    return c

## Assemble constraint
def con(ztraj, x_init):
    c = jnp.zeros(m_nlp)
    xtraj, _ = build_traj(ztraj)
    c = c.at[c_init_inds].set(ztraj[:Nx] - x_init) # Initial constraint
    c = c + dynamics_constraint(ztraj)
    c = c + state_constraint(ztraj)
    c = c + control_constraint(ztraj)
    return c

## Compile & auto differentiate
con_jit = jax.jit(con)
con_jacobian_jit = jax.jit(jax.jacfwd(con, argnums=0))



# %% LEVERAGE SPARSITY

x_init_zero = np.zeros((Nx,))

def get_sparse_jacobian_structure():
    'Function to get non-zero index locations of constraint jacobian, necessary since con_jacobian_jit is a function'
    key = jax.random.PRNGKey(0)
    z0 = jax.random.normal(key, shape=(n_nlp,))

    rows, cols = np.nonzero(con_jacobian_jit(z0, x_init_zero))
    return rows, cols

rows, cols = get_sparse_jacobian_structure()


# Hessian sparsity (symmetric)
H = hessian_cost(np.zeros(n_nlp), 0) # Can evaluate at zero since second deriv of x.T@Q@x generally constant Q
hess_rows, hess_cols = np.nonzero(np.triu(H) != 0)



# %% SOLVE PROBLEM

## cyipopt problem class
class TrajectoryOptimizationProblem(object):
    def __init__(self, n, m, rows, cols):
        self.n = n  # decision variable dimension
        self.m = m  # constraint dimension
        self.rows = rows
        self.cols = cols
        self.i_offset = 0  # progression along path at each MPC step
        self.x_init = jnp.zeros(Nx)  # will be updated at each timestep

    def objective(self, x):
        return cost_jit(x, self.i_offset)

    def gradient(self, x):
        return cost_gradient_jit(x, self.i_offset).flatten()

    def constraints(self, x):
        return con_jit(x, self.x_init).flatten()

    def jacobian(self, x):
        J = con_jacobian_jit(x, self.x_init)
        return J[self.rows,self.cols].flatten()

    def jacobianstructure(self):
        return (self.rows, self.cols)

    def hessianstructure(self):
        # rows, cols = np.nonzero(np.ones((self.n, self.n)))
        return (hess_rows, hess_cols)

    def hessian(self, x, lagrange, obj_factor):
        H = hessian_cost(x)
        return H[hess_rows, hess_cols].flatten()

## Set bounds and initial guess
lb = -np.inf * np.ones(n_nlp)
ub = np.inf * np.ones(n_nlp)

## State guess: noise around xref
key = jr.PRNGKey(0) # Seed
key, sub = jr.split(key)
xguess = xref_arr[:, :Nt_h] + 0.0001 * jr.normal(sub, shape=xref_arr[:, :Nt_h].shape)

## Control guess: tile across Nt-1 and noise
key, sub = jr.split(key)
uguess = uref_arr[:,:(Nt_h-1)] + 0.0001 * jr.normal(sub, shape=(uref_arr[:,:(Nt_h-1)]).shape) + Fmto*0.5 # drop last column of uref since controls only go to Nt-1

z_mat = np.hstack([xguess[:,:-1].T, uguess.T])  # shape (Nt-1, Nx+Nu)
z0 = np.concatenate([z_mat.ravel(), xguess[:,-1]])

print("initial max constraint resid:", float(jnp.max(jnp.abs(con_jit(z0, x_init_zero)))))

## Initiate Problem
problem = TrajectoryOptimizationProblem(n=n_nlp, m=m_nlp, rows=rows, cols=cols)
nlp = cyipopt.Problem(
    n=n_nlp,
    m=m_nlp,
    problem_obj=problem,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu
)



nlp.add_option("max_iter", 20)
nlp.add_option("tol", 1e-2)
nlp.add_option("constr_viol_tol", 1e-5)
nlp.add_option("hessian_approximation", "limited-memory")
nlp.add_option("mu_strategy", "adaptive")
nlp.add_option("print_level", 0) # Decreased number as speed is important





# %% RUN MUJOCO SIM

## Setup sim recording variables
i = 0
x_hist = np.zeros_like(xref)
u_hist = np.zeros_like(uref)

## Make sure JAX is compiling ahead of the loop
dummy_z = jnp.zeros(n_nlp)
dummy_x = jnp.zeros(Nx)
_ = cost_jit(dummy_z, 0)
_ = cost_gradient_jit(dummy_z, 0)
_ = con_jit(dummy_z, dummy_x)
_ = con_jacobian_jit(dummy_z, dummy_x)


## MPC warm starting function
def next_z0(z_sol):
    z0_new = jnp.concatenate([
        z_sol[(Nx + Nu):], # remove first x and u
        z_sol[-(Nx + Nu):], # repeat last state and control
    ])
    return np.array(z0_new)


pos_offset = [0.4276056667, -0.4276056667]
# Open the MuJoCo viewer
# with viewer.launch_passive(model, data) as sim_viewer:
## Run simulation
for i in range(Nt*loops):
# while sim_viewer.is_running():
    ## Current joint positions and velocities
    qpos = data.qpos[[0, 2]] + pos_offset
    qpos = [np.pi/2+qpos[1], np.pi/2+qpos[0]]
    qvel = data.qvel[[0, 2]]
    qvel = [qvel[1], qvel[0]]
    cur_x = jnp.hstack((qpos, qvel))
    problem.i_offset = i  # tell cost which slice of xref to use
    problem.x_init = cur_x
    
    ## Run MPC
    z_sol, _ = nlp.solve(z0)
    x_sol, u_sol = build_traj(z_sol)
    F_r = u_sol[0, 0]
    F_l = u_sol[0, 1]
    
    ## Calculate system positions
    theta_r = np.pi/2 - qpos[0]
    theta_l = qpos[1] - np.pi/2
    l_muscle_r = angle_to_muscle_length(theta_r)
    l_muscle_l = angle_to_muscle_length(theta_l)
    
    torque = np.array([muscle_force_to_moment(F_l, l_muscle_l, theta=theta_l),
                       -muscle_force_to_moment(F_r, l_muscle_r, theta=theta_r)])
    
    ## Apply torques to actuators
    data.ctrl[:] = torque
    
    ## Record current step
    x_hist[i%Nt,:] = cur_x
    u_hist[i%Nt,:] = np.array([F_r, F_l])

    ## Step simulation
    mujoco.mj_step(model, data)
    # sim_viewer.sync()

    z0 = next_z0(z_sol)

    ## Control simulation speed (real-time)
    # time.sleep(model.opt.timestep)
    # i += 1



# %% PLOTTING

plt.plot(x_hist[:,0], x_hist[:,1], color='C0')
plt.plot(xref[:,0], xref[:,1], color='orange', linestyle='dashed')

hist_time = np.linspace(0, t_final, num=np.shape(xref)[0])
plt.figure()
plt.plot(hist_time, x_hist[:, 0], color='blue')
plt.plot(hist_time, xref[:, 0], color='blue', linestyle='dashed')
plt.plot(hist_time, x_hist[:, 1], color='orange')
plt.plot(hist_time, xref[:, 1], color='orange', linestyle='dashed')
plt.ylabel('Arm Angle (rad)')
plt.xlabel('Time (sec)')
plt.title('Desired (dashed) vs Actual (solid) Arm Angles')

plt.figure()
plt.plot(hist_time, u_hist[:, 0], color='blue')
plt.plot(hist_time, u_hist[:, 1], color='orange')
plt.title('control force history (N)')

plt.figure()
ee_act = forward_kinematics(x_hist.T).T
plt.plot(ee_act[:,0], ee_act[:,1], color='C0')
plt.plot(ee_ref[:,0], ee_ref[:,1], color='orange', linestyle='dashed')
plt.gca().set_aspect('equal')
plt.title('End Effector Position (m)')

# %% Save CSV

# import os
# base_dir = os.path.dirname(os.path.abspath(__file__))
# csv_path = os.path.join(base_dir, "misc", "mujoco_MPC_forces_small.csv")
# np.savetxt(csv_path, np.hstack([x_hist, u_hist]), delimiter=",", header="theta_r,theta_l,dtheta_r,dtheta_l,ctrl_r,ctrl_l", comments="")

