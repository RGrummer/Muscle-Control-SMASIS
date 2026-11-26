import cyipopt
import matplotlib.pyplot as plt
import numpy as np
from raw_i3_force import I3MuscleForce
import jax
import jax.numpy as jnp
import jax.random as jr
import os
jax.config.update("jax_enable_x64", True)

### Controller for contracting an I3 muscle for a specific amount of force over time
### State x = [aprime] unclipped activation
### Control u = [f_stim] stimulation frequency
### Feed in Lmt (total lenght of the muscle CE and SEE) vs time and desired force vs time

## Simulation Variables
Nx = 1 # Number of states
Nu = 1 # Number of controls
t_final = 30.0 # s, Time length of sim
dt = 0.05 # s, Timestep of sim
Nt = int(t_final/dt) # Number of timesteps
n_nlp = Nx * Nt + Nu * (Nt - 1)

## Cost matrices
Q = 5e2 # Force tracking cost
R = 5e-3 # Control cost
S = 2e-1 # Smoothing cost
Qf = Q # Terminal cost = same as Q
u_baseline = 20.0 # baseline u to have guess around

## I3 value estimates based on Sukhnandan et al. (2024)
i3 = dict(
    A = np.array([1.55, 7.90, 0.33, 8.33]),
    B = np.array([7.44, -29.74, 37.17, -13.87]),
    C = np.array([-1.89, 1.70, 0.75, 0.55]), # Coefficients for parallel elastic element force-length relationship
    D = np.array([-0.77, 0.07, 0.35, 0.07]),
    E = np.array([0.99, 13.03, 1.37, 2.32, -5.71]),
    Y = np.array([5.44, -21.15, 25.16, -8.49]),
    beta = 0.15,
    tau = 0.60,
    a0 = 0.74,
    g = 3.85,
    KT = 11.95,
    L0 = 0.01602, # m, rest length of I3
    Lmto = 0.01803, # m, length of max I3 force
    Fmto = 1.61, # N, max force of I3 at Lmto
)
i3["passive_lmt"] = (i3["C"][3] + np.log(-i3["C"][0]/i3["C"][1]))/i3["C"][2] # Muscle length at which parallel elastic element starts to take effect
i3["slackrate"] = -(i3["E"][0]*np.exp(i3["E"][1]*(0 - i3["E"][2])) - i3["E"][3]*np.exp(i3["E"][4]*0)) # -IFV with FV=0

# Convert i3 to all JAX arrays
i3 = {k: jnp.asarray(v, dtype=jnp.float64) if isinstance(v, (np.ndarray, list, tuple)) else v for k, v in i3.items()}

## Muscle limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = i3["Lmto"]*(1+percent_L_range) # Max I3 operation length
L_min = i3["Lmto"]*(1-percent_L_range) # Min I3 operation length

## Control limits
f_stim_min = 0

# %% REFERENCE F AND Lmt PATHS FOR TESTING

def reference_trajectory():
    'Develop a reference trajectory for the linkage'
    y_traj = jnp.ones([Nt]) * 0.5 # Constant force
    Lmt_traj = jnp.ones([Nt]) * 0.013 # Constant length
    dLmt_traj = Lmt_traj[1:] - Lmt_traj[:-1]
    dLmt_traj = jnp.append(dLmt_traj, dLmt_traj[-1])
    
    return y_traj, Lmt_traj, dLmt_traj

## Morphology variables
delta_x = 1.853e-02
delta_y = 5.375e-03
r_muscle_attach = 9.572e-03

## Calculated morphology variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

def angle_to_muscle_length_velocity(theta, dtheta):
    'Finds muscle length and velocity given specific angle of actuated arm'
    # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
    muscle_length = np.sqrt(delta_x**2 + delta_y**2 + r_muscle_attach**2
                             + 2*delta_y*r_muscle_attach*np.cos(theta)
                             - 2*delta_x*r_muscle_attach*np.sin(theta))
    muscle_velocity = -(delta_x*np.cos(theta) + delta_y*np.sin(theta))*r_muscle_attach*dtheta/muscle_length
    return muscle_length, muscle_velocity

def csv_ref(csv_name, side_num, end_int = -5, start_int = 10):
    # side_num = 0 for left side and = 1 for right side
    import pandas as pd
    traj = np.array(pd.read_csv(csv_name))
    
    # Define the start and end points for interpolation, allows for smoother looping
    num_int = start_int - end_int - 1
    interp_rows = np.linspace(traj[end_int], traj[start_int], num=num_int+2, axis=0) 
    for i in range(end_int+1, start_int):
        traj[i] = interp_rows[i-end_int]
    
    F_traj = traj[:, 4+side_num]
    theta_traj = traj[:, 0+side_num]
    dtheta_traj = traj[:, 2+side_num]
    if side_num == 0:
        theta_traj = np.pi/2 - theta_traj
        dtheta_traj = -dtheta_traj
    else:
        theta_traj = theta_traj - np.pi/2
    Lmt_traj, Vmt_traj = angle_to_muscle_length_velocity(theta_traj, dtheta_traj)
    
    # # Doubles trajectory so you have two loops
    F_traj=jnp.hstack([F_traj, F_traj])
    Lmt_traj=jnp.hstack([Lmt_traj, Lmt_traj])
    Vmt_traj=jnp.hstack([Vmt_traj, Vmt_traj])
    
    ratio = int(dt/0.01)
    F_traj = F_traj.reshape(-1, ratio, 1).mean(axis=1) # Averages every couple values together to improve solve speed
    Lmt_traj = Lmt_traj.reshape(-1, ratio, 1).mean(axis=1)
    Vmt_traj = Vmt_traj.reshape(-1, ratio, 1).mean(axis=1)
    
    F_traj = F_traj.reshape(-1)
    Lmt_traj = Lmt_traj.reshape(-1)
    Vmt_traj = Vmt_traj.reshape(-1)
    
    return F_traj, Lmt_traj, Vmt_traj
    

# F_ref, Lmt_ref, Vmt_ref = reference_trajectory() # [1, Nt] arrays
F_ref, Lmt_ref, Vmt_ref = csv_ref(os.path.join("misc", "mujoco_MPC_forces_small_heart.csv"), 1) # [1, Nt] arrays, index of 0 for left side or 1 for right side
x_init = jnp.array([0.0])

plt.plot(np.linspace(0, t_final, num=Nt), Lmt_ref)


# %% MUSCLE FORCE FUNCS

def muscle_activation_deriv(aprime, fstim, params):
    'Given control input and state outputs daprime, derivative of unscaled, unthresholded activation'
    uf = params["A"][0]/(params["A"][0] + 
                    params["A"][1]*jnp.exp(-params["A"][2]*(fstim - params["A"][3])))
    daprime = 1/params["tau"]*(uf - (params["beta"] + (1 - params["beta"])*uf)*aprime)
    return daprime

def no_passive(arg):
    'Function for if the muscle has no passive force'
    return jnp.array(0.0, dtype=jnp.float64)

def yes_passive(arg):
    'Function for if the muscle is extended enough to have passive force'
    'input arg = [lmt, muscle_params]'
    params = arg[1]
    fPE = params["C"][0] + params["C"][1]*jnp.exp(params["C"][2]*arg[0]-params["C"][3])
    return fPE

def force_length(lmt, params):
    'Force length relationship for full I3 muscle'
    LT = params["B"][0]*lmt**3 + params["B"][1]*lmt**2 + params["B"][2]*lmt + params["B"][3]
    return LT

def force_negative_velocity(arg):
    'Force-velocity relationship of the muscle if muscle velocity < 0'
    'input arg = [vmt, muscle_params]'
    params = arg[1]
    FV = 1 + params["D"][0]/(1 - params["D"][1]/arg[0])
    return FV

def force_positive_velocity(arg):
    'Force-velocity relationship of the muscle if muscle velocity > 0'
    'input arg = [vmt, muscle_params]'
    params = arg[1]
    FV = 1 + params["D"][2]/(1 + params["D"][3]/arg[0])
    return FV

def force_zero_velocity(arg):
    'Force-velocity relationship of the muscle if muscle velocity = 0'
    'input arg = [vmt, muscle_params]'
    return 1.0

def force_nonzero_velocity(arg):
    'Force-velocity relationship of the muscle if muscle velocity != 0'
    'input arg = [vmt, muscle_params]'
    FV = jax.lax.cond(arg[0] < 0.0,
                      force_negative_velocity,
                      force_positive_velocity,
                      operand=(arg)
                      )
    return FV

def muscle_force(aprime, Lmt, Vmt, params):
    'Just calculate muscle force'
    lmt = Lmt/params["Lmto"] # Normalized muscle length
    vmt = Vmt/params["Lmto"] # Normalized muscle velocity
    
    ## Passive muscle force
    fPE = jax.lax.cond(lmt < params["passive_lmt"],
                       no_passive,
                       yes_passive,
                       operand=(lmt, params)
                       )
    
    ## Active muscle force
    FL = force_length(lmt, params)
    FV = jax.lax.cond(vmt == 0.0,
                      force_zero_velocity,
                      force_nonzero_velocity,
                      operand=(vmt, params)
                      )
    
    ## Activation scaling and thresholding
    a = jnp.clip(params["g"]*(aprime - params["a0"]), 0.0, 1.0)
    
    ## Active contractile element force
    fCE = a*FV*FL
    
    ## Total muscle force
    Fmt = params["Fmto"]*(fCE + fPE)
    
    return Fmt


# %% SYSTEM DYNAMICS

def dynamics(x, u, Lmt):
    'Calculate derivative of state using lagrange mechanics given specific x and u'
    daprime = muscle_activation_deriv(x, u, i3)
    # dx = jnp.array([daprime])
    return daprime #dx


def rk4_dynamics(x, u, Lmt, dt):
    'Use Runga-Kutta 4 to approximate state change over one timestep'
    f = lambda s: dynamics(s, u, Lmt)
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    
    x_next = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    return x_next



# %% INITIAL GUESSES AND REFS

## Control reference: 0 to minimize stimulation applied
u_ref = jnp.zeros([Nt])

## State guess with noise
x_guess = jnp.zeros([Nt])
# x_guess = x_guess.at[1, :].set(F_ref[0, :]/i3["KT"] - Lmt_ref[0, :])
key = jr.PRNGKey(0) # Seed
key, sub = jr.split(key)
# x_guess = x_guess + 0.0001 * jnp.abs(jr.normal(sub, shape=x_guess.shape))

## Control guess: tile across Nt-1 and noise
key, sub = jr.split(key)
u_guess = u_ref[:-1] + u_baseline + 1.0 * jr.normal(sub, shape=(u_ref[:-1]).shape) # drop last column of uref since controls only go to Nt-1


# Prepare rollout array (Nt)
x_roll = jnp.zeros((Nt,))
x_roll = x_roll.at[0].set(x_init[0])

for s in range(Nt - 1):
    x_next = rk4_dynamics(x_roll[s], u_guess[s], Lmt_ref[s], dt)
    x_roll = x_roll.at[s + 1].set(x_next)

# Use x_roll as initial guess (add tiny noise if you want)
x_guess = x_roll + 1e-8 * jnp.abs(jr.normal(jr.PRNGKey(1), shape=x_roll.shape))

# Rebuild z0 for Ipopt
z_mat = np.vstack([np.array(x_guess[:-1]), np.array(u_guess)]).T  # convert to numpy
z0 = np.concatenate([z_mat.ravel(), np.array([x_guess[-1],])])


# %% COST FUNCTIONS AND INDEXING UTILITIES

def stage_cost(x, u, k):
    F = muscle_force(x, Lmt_ref[k+1], Vmt_ref[k+1], i3)
    deltay = F - F_ref[k+1] # Difference between actual and desired muscle force
    deltau = u # - u_baseline # u_baseline can help keep the stimulation from staying at 0
    return 0.5 * deltay*Q*deltay + 0.5 * deltau*R*deltau

def terminal_cost(x):
    F = muscle_force(x, Lmt_ref[-1], Vmt_ref[-1], i3)
    deltay = F - F_ref[-1]
    return 0.5 * deltay*Qf*deltay

def smoothing_cost(u_traj): # Punishes changes in u to smooth outputs since otherwise it gets very noisy
    du = u_traj[1:] - u_traj[:-1]
    return 0.5 * jnp.sum(du * S * du)

def cost(ztraj):#, F_ref): # ztraj is 1D vector of [x0, u0, x1, u1, ... , uN-1, xN]
    x_traj, u_traj = build_traj(ztraj)

    # vectorize stage_cost over time indices 0…Nt-2
    stage_indices = jnp.arange(Nt-1)
    J_stages = jnp.sum(jax.vmap(stage_cost, in_axes=(0,0,0))(x_traj[1:], u_traj, stage_indices))
    return J_stages + terminal_cost(x_traj[-1]) + smoothing_cost(u_traj)


def unpack(ztraj):
    """
    Return:
      z_mat   -- shape (Nt-1, Nx+Nu), each row = [x_k, u_k]
      x_term  -- shape (Nx,), terminal state
    """
    flat_len = (Nt - 1) * (Nx + Nu)
    z_mat = jnp.reshape(ztraj[:flat_len], (Nt - 1, Nx + Nu))
    x_term = ztraj[flat_len:]
    return z_mat, x_term

def build_traj(ztraj):
    z_mat, x_term = unpack(ztraj)
    x_traj = jnp.vstack([z_mat[:, :Nx], x_term[jnp.newaxis, :]]).ravel()  # shape (Nt,)
    u_traj = z_mat[:, Nx:].ravel()  # shape (Nt-1,)
    return x_traj, u_traj

## Compile & auto-differentiate
cost_jit = jax.jit(cost)
cost_gradient_jit = jax.jit(jax.jacfwd(cost))
hessian_cost = jax.jit(jax.hessian(cost))


# %% CONSTRAINT FUNCTIONS
### Constraint functions work by you having them output values and later creating upper and lower bound matrices for them

## Set up constraint arrays
init_end = 0 + Nx
dyn_end = init_end + Nx*(Nt - 1)
control_end = dyn_end + (Nt - 1)
m_nlp = control_end # number of constraints

c_init_inds    = jnp.arange(0, init_end)
c_dyn_inds     = jnp.arange(init_end, dyn_end)
c_control_inds = jnp.arange(dyn_end, control_end)

## Constrain array lower and upper bounds
cl = np.zeros(m_nlp, dtype=np.float64) # Lower constraint bound
cu = np.zeros(m_nlp, dtype=np.float64) # Upper constraint bound
cl[c_control_inds] = 0.0 # f_stim_min
cu[c_control_inds] = 40.0 # f_stim_max

def dynamics_constraint(ztraj):
    x_traj, u_traj = build_traj(ztraj)
    
    d_init = jnp.zeros((Nt - 1))
    
    def single_dynamics_inst(s, d):
        x_s = x_traj[s]
        u_s = u_traj[s]
        
        fx = rk4_dynamics(x_s, u_s, Lmt_ref[s], dt)
        df = fx - x_traj[s + 1]
        return d.at[s].set(df) # this is how you update a number or matrix in a loop in Jax
    
    # This function encodes a loop in jax
    d = jax.lax.fori_loop(
        0, # starting index
        Nt - 1, # ending index
        single_dynamics_inst, # function to loop over. It accepts an index (s) and a vector d that is the same shape as d_init, which is the thing that we modify in the loop
        d_init) # vector with the same shape as what we return from the loop

    # Stuff into constraint vector
    c = jnp.zeros(m_nlp)
    c = c.at[c_dyn_inds].set(jnp.reshape(d, (-1,)))
    return c

def control_constraint(ztraj):
    _, u_traj = build_traj(ztraj)
    
    us = jnp.zeros((Nt-1))
    def get_u(i, u_vec):
        u = u_traj[i]
        u_vec = u_vec.at[i].set(u)
        return u_vec
    us = jax.lax.fori_loop(0, Nt-1, get_u, us)
    
    c = jnp.zeros(m_nlp)
    c = c.at[c_control_inds].set(us)
    return c


## Assemble constraint
def con(ztraj):
    c = jnp.zeros(m_nlp)
    xtraj, _ = build_traj(ztraj)
    c = c.at[c_init_inds].set(ztraj[:Nx] - x_init) # Initial constraint
    c = c + dynamics_constraint(ztraj)
    c = c + control_constraint(ztraj)
    return c


## Compile & auto differentiate
con_jit = jax.jit(con)
con_jacobian_jit = jax.jit(jax.jacfwd(con))

print("max |con(z0)| (numpy):", np.max(np.abs(np.array(con_jit(z0)))))


# %% LEVERAGE SPARSITY

def get_sparse_jacobian_structure():
    'Function to get non-zero index locations of constraint jacobian, necessary since con_jacobian_jit is a function'
    key = jax.random.PRNGKey(0)
    z0 = jax.random.normal(key, shape=(n_nlp,))

    rows, cols = np.nonzero(con_jacobian_jit(z0))
    return rows, cols

rows, cols = get_sparse_jacobian_structure()


# Hessian sparsity (symmetric)
H = np.asarray(hessian_cost(np.zeros(n_nlp))).squeeze() # Can evaluate at zero since second deriv of x.T@Q@x generally constant Q
hess_rows, hess_cols = np.nonzero(np.triu(H) != 0)



# %% SOLVE PROBLEM

## cyipopt problem class
class TrajectoryOptimizationProblem(object):
    def __init__(self, n, m, rows, cols):
        self.n = n  # decision variable dimension
        self.m = m  # constraint dimension
        self.rows = rows
        self.cols = cols

    def objective(self, x):
        return cost_jit(x)#, self.F_ref)

    def gradient(self, x):
        return cost_gradient_jit(x).flatten()

    def constraints(self, x):
        return con_jit(x).flatten()

    def jacobian(self, x):
        J = con_jacobian_jit(x)
        
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

# z_mat = np.hstack([x_guess[:,:-1].T, u_guess.T])  # shape (Nt-1, Nx+Nu)
# z0 = np.concatenate([z_mat.ravel(), x_guess[:,-1]])

print("initial max constraint resid:", float(jnp.max(jnp.abs(con_jit(z0)))))

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
# nlp.F_ref = F_ref_new

nlp.add_option("max_iter", 1500)
nlp.add_option("tol", 1e-4)
nlp.add_option("constr_viol_tol", 1e-6)
nlp.add_option("hessian_approximation", "limited-memory")
nlp.add_option("print_level", 4)


z_sol, info = nlp.solve(z0)


# %% PLOTTING RESULT

x_sol, u_sol = build_traj(z_sol)
a_sol = jnp.clip(i3["g"]*(x_sol[:] - i3["a0"]), 0.0, 1.0)
F_sol = np.zeros_like(F_ref)
FPE_sol = np.zeros_like(F_ref)
for i in range(Nt):
    F_sol[i] = muscle_force(x_sol[i], Lmt_ref[i], Vmt_ref[i], i3)
    lmt = Lmt_ref[i]/i3["Lmto"]
    FPE_sol[i] = i3["Fmto"]*jax.lax.cond(lmt < i3["passive_lmt"],
                       no_passive,
                       yes_passive,
                       operand=(lmt, i3)
                       )

timeu = np.linspace(0, t_final, num=Nt-1)
timex = np.linspace(0, t_final, num=Nt)

## State Plot
figF, axF = plt.subplots()
axF.plot(timex, F_sol, color='C0', label='F_sol')
# axF.plot(timex, FPE_sol, color='purple', linestyle='--')
axF.plot(timex, F_ref, color='orange', linestyle='--', label='F_ref')
axF.set_title('Solution vs reference force')
axF.set_ylabel('Muscle Force (N)')
axF.set_xlabel('Time (s)')
axF.legend()
# axF.grid()
plt.show()

## State Plot
figa, axa = plt.subplots()
axa.plot(timex, Lmt_ref/i3["Lmto"], color='orange', label='lmt')
axa.plot(timex, Vmt_ref/i3["Lmto"], color='purple', label='Vmt')
axa.plot(timex, x_sol[:], color='C0', label='aprime')
axa.plot(timex, a_sol, color='green', label='a')
axa.plot(timex, x_guess[:], color='red', linestyle='--', label='aprime guess')
axa.grid()
axa.legend()
plt.show()

# Controls plot
figu, axu = plt.subplots()
axu.plot(timeu, u_sol[:])
axu.set_title('Applied Stimulation')
axu.set_ylabel('Stimulation frequency (Hz)')
axu.set_xlabel('Time (s)')
# axu.grid()
plt.show()


# %% COMPARE FULL HILL FORCE

muscle = I3MuscleForce(dt)
full_force_hist = np.zeros((int(t_final/dt),))
full_force_hist[0] = muscle.calc_force(Lmt_ref[0])
for i in range(0, int(t_final/dt)-1):
    full_force_hist[i+1] = muscle.apply_forces(u_sol[i], Lmt_ref[i+1])

# Comparison plot
figf, axf = plt.subplots()
axf.plot(timex, F_sol, color='C0')
axf.plot(timex, full_force_hist, color='orange', linestyle='dashed')
axf.set_title('Full hill force (orange) vs simple')
plt.show()


# %% SAVE CSV
'Run MPC once for left side (0) and do "u_sol_l = u_sol", run once for right side (1) and do "u_sol_r = u_sol" then run this section block'
'MAKE SURE CSV FILE NAME IS WHAT YOU WANT BEFORE RUNNING'

# base_dir = os.path.dirname(os.path.abspath(__file__))
# csv_path = os.path.join(base_dir, "misc", "MPC_fstims_heart.csv")
# np.savetxt(csv_path, np.array([u_sol_l, u_sol_r]).T, delimiter=",", header="fstim_l, fstim_r", comments="")

