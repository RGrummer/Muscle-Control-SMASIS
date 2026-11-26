import mujoco
from mujoco import viewer
import time
import numpy as np
import matplotlib.pyplot as plt
from raw_i3_force import I3MuscleForce
import pandas as pd
import os

# Load model
model = mujoco.MjModel.from_xml_path("proper_mujoco_model/robot.xml")
data = mujoco.MjData(model)

t_final = 15.0 # s, Time length of sim
dt = model.opt.timestep # s, Timestep of sim
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

## Calculated morphology variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

L0 = .016 # m, rest length of I3
muscle_r = I3MuscleForce(dt)
muscle_l = I3MuscleForce(dt)

## Manipulator values
l1 = 0.153
l2 = 0.1157
angle_offset = np.deg2rad(21.8438) # rad, offset of encoder angle readings off of straight ahead

# %% TRAJ AND HELPERS

def angle_to_muscle_length_velocity(theta, dtheta):
    'Finds muscle length and velocity given specific angle of actuated arm'
    # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
    muscle_length = np.sqrt(delta_x**2 + delta_y**2 + r_muscle_attach**2
                             + 2*delta_y*r_muscle_attach*np.cos(theta)
                             - 2*delta_x*r_muscle_attach*np.sin(theta))
    muscle_velocity = -(delta_x*np.cos(theta) + delta_y*np.sin(theta))*r_muscle_attach*dtheta/muscle_length
    return muscle_length, muscle_velocity

def angle_to_muscle_length(theta):
    'Finds muscle length given specific angle of actuated arm'
    # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
    muscle_length = np.sqrt(delta_x**2 + delta_y**2 + r_muscle_attach**2
                             + 2*delta_y*r_muscle_attach*np.cos(theta)
                             - 2*delta_x*r_muscle_attach*np.sin(theta))
    return muscle_length

def inverse_kinematics(p):
    'Calculates IK of 5 bar linkage for actuated arm angles giveen end effector position p=[x,y]'
    acos_term = np.arccos((p[0]**2 + p[1]**2 + l1**2 - l2**2)/(2*l1*np.sqrt(p[0]**2 + p[1]**2)))
    atan_term = np.arctan(p[0]/p[1])
    # theta_r = np.pi/2 - (acos_term + atan_term) # Pi terms appear there because this math was originally done for the "relative" thetas off vertical
    # theta_l = (acos_term - atan_term) + np.pi/2
    theta_r = -(acos_term + atan_term)
    theta_l = (acos_term - atan_term)
    return np.array([theta_l, theta_r])

def forward_kinematics(x):
    'Calculate forward kinematics of 5 bar linkage end effector x and y positions'
    x = x + np.pi/2
    mid = np.array([l1/2*(np.cos(x[0]) + np.cos(x[1])), 
                     l1/2*(np.sin(x[0]) + np.sin(x[1]))]) # Midpoint between actuated arms
    avg_angle = (x[0] + x[1])/2
    h = np.sqrt(l2**2 - l1**2/4*(2 - 2*np.cos(x[0] - x[1]))) # Equation for dist from line between actuated arm ends to end effector
    e_perp = np.array([np.cos(avg_angle), np.sin(avg_angle)])
    endpoint = mid + h * e_perp # Two possible intersection points, choose higher "up" point
    return endpoint

def circle_reference_trajectory():
    'Circle reference trajectory for end effector'
    # NOTE: Should not have arm angle less than 0.48 rad relative
    center = np.array([0.0, 0.19])  # m
    radius = 0.025 # m
    return np.array([center + radius * np.array([np.cos(2*np.pi*i), np.sin(2*np.pi*i)])
        for i in np.linspace(0, 1, num=Nt)]) # Returns an [Nt, 2] array

def heart_reference_trajectory():
    'Heart shaped reference trajectory for end effector'
    center = np.array([0.0, 0.18])  # m
    scale = 0.0018  # scales to about 0.025 m width/height

    heart_points = np.array([
        center + scale * np.array([
            16 * np.sin(i)**3,
            13 * np.cos(i) - 5 * np.cos(2*i) - 2 * np.cos(3*i) - np.cos(4*i)
        ])
        for i in np.linspace(0, 2*np.pi, num=Nt)
    ])
    return heart_points # Returns an [Nt, 2] array

def angle_trajectory(end_traj_func):
        'Calculates angle trajectory of a reference end effector trajectory'
        end_traj = end_traj_func()
        angle_traj = np.zeros_like(end_traj)
        for i in range(np.shape(end_traj)[0]):
            angle_traj[i,:] = inverse_kinematics(end_traj[i,:])
        return angle_traj # Returns an [Nt, 2] array

def muscle_length_to_angle(muscle_length):
    'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
    theta = np.pi/2 + phi - np.arccos((r_muscle_attach**2 + r_OC**2 - muscle_length**2)/(2*r_muscle_attach*r_OC))
    return theta

def muscle_force_to_moment(F, muscle_length, theta = None):
    'Converts force from muscle to moment on the respective arm'
    'MUST provide either muscle_length or theta'
    if theta is None:
        theta = muscle_length_to_angle(muscle_length)
    M = r_muscle_attach*F*(delta_x*np.cos(theta) + delta_y*np.sin(theta))/muscle_length
    return M

# Target joint positions (in radians)
# ref_traj = circle_reference_trajectory # reference end effector path
ref_traj = heart_reference_trajectory # reference end effector path
ee_ref = ref_traj()
target_pos_arr = angle_trajectory(ref_traj)



# %% SETUP CONTROLS

## PID control parameters for full PID control
# kp_f = 14e1 # proportional gains for 2 motors
# kd_f = 17e0 # derivative gains
# ki_f = 23e0 # integral gains
kp_f = 12e1 # proportional gains for 2 motors
kd_f = 18e0 # derivative gains
ki_f = 22e0 # integral gains

## PID control parameters for higher level PID control for SIM
# kp_h = 4.0e1  # proportional gains
# kd_h = 15.0e-1  # derivative gains
# ki_h = 15.0e-1 # integral gains

## PID control parameters for higher level PID control for ID
kp_h = 15.0e0  # proportional gains
kd_h = 10.0e-1  # derivative gains
ki_h = 35.0e-1 # integral gains

## Simple ID time parameter
tau_aprime_simple = 3.0

## Full ID time parameters
tau_lm = 1.1
tau_aprime = 1.0

def full_PID_control(params):
    'params = [pos_error, vel, integral]'
    fstim = np.clip(kp_f*(params[0]) + ki_f*params[2] - kd_f*params[1], 0.000001, 40.0)
    return fstim

def higher_PID_control(params):
    'Higher level PID control code, returns desired force for use with ID controller'
    'params = [pos_error, vel, integral]'
    force_d = np.clip(kp_h*params[0] + ki_h*params[2] - kd_h*params[1], 0.000001, 1.61)
    return force_d

def simple_SIM_control(params):
    'Simple Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
    Lmt, Vmt = angle_to_muscle_length_velocity(params[3], params[1])
    
    ## Calculate main simple hill variables
    muscle = params[4]
    f_d = np.clip(F_d/muscle.Fmto, 0.0, 1.0) # desired muscle force
    lmt = Lmt/muscle.Lmto
    vmt = Vmt/muscle.Lmto
    fPE = muscle.passive(lmt)
    FV = muscle.force_velocity(vmt)
    LT = muscle.total_force_length(lmt)
    
    ## Calculate stimulus amount
    a_d = np.clip((f_d - fPE)/(FV*LT), 0.0, 1.0)
    aprime_d = a_d/muscle.g + muscle.a0 # desired aprime unscaled activation

    daprime_d = (aprime_d - muscle.aprime)/tau_aprime_simple
    uf_d = (muscle.tau*daprime_d + muscle.beta*muscle.aprime)/(1 - muscle.aprime*(1-muscle.beta))
    fstim_d = muscle.A[3] - np.log(np.max([(muscle.A[0]/uf_d - muscle.A[0])/muscle.A[1], 1e-8]))/muscle.A[2]
    
    _ = muscle.apply_forces(fstim_d, Lmt) # updates virtual muscle.aprime
    
    return fstim_d

def full_SIM_control(params):
    'Full Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
    Lmt = angle_to_muscle_length(params[3])
    
    ## Calculate main simple hill variables
    muscle = params[4]
    f_d = np.clip(F_d/muscle.Fmto, 0.0, 1.0)
    lmt = Lmt/muscle.Lmto
    fPE = muscle.passive(lmt)
    lm_d = lmt - (f_d - fPE)/muscle.KT # desired lm length of Contractile Element
    
    dlm_d = (lm_d - muscle.lm)/tau_lm
    FV_d = muscle.force_velocity(dlm_d)
    LT = muscle.force_length(muscle.lm)
    a_d = np.clip((f_d - fPE)/(FV_d*LT), 0.0, 1.0)
    aprime_d = a_d/muscle.g + muscle.a0 # desired aprime unscaled activation
    
    daprime_d = (aprime_d - muscle.aprime)/tau_aprime
    uf_d = (muscle.tau*daprime_d + muscle.beta*muscle.aprime)/(1 - muscle.aprime*(1-muscle.beta))
    fstim_d = muscle.A[3] - np.log(np.max([(muscle.A[0]/uf_d - muscle.A[0])/muscle.A[1], 1e-8]))/muscle.A[2]
    fstim_d = np.clip(fstim_d, 0.0, 40.0)
    
    return fstim_d

def ID_control(params):
    'Inverse dynamics lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
    Lmt, Vmt = angle_to_muscle_length_velocity(params[3], params[1])
    
    ## Calculate main simple hill variables
    muscle = params[4]
    lmt = Lmt/muscle.Lmto
    vmt = Vmt/muscle.Lmto
    FV = muscle.force_velocity(vmt)
    LT = muscle.total_force_length(lmt)
    F_cur = muscle.calc_force(lmt) # current muscle force
    
    ## Calculate stimulus amount
    f_cur = np.clip(F_cur/muscle.Fmto, 0.0, 1.0) # current muscle force
    f_d = np.clip(F_d/muscle.Fmto, 0.0, 1.0) # desired muscle force
    da_d = (f_d - f_cur)/(FV*LT*dt)
    
    daprime_d = da_d/muscle.g # desired change in aprime unscaled activation
    uf_d = np.max([(muscle.tau*daprime_d + muscle.beta*muscle.aprime)/(1 - muscle.aprime*(1-muscle.beta)), 0.0000001])
    fstim_d = muscle.A[3] - np.log(np.max([(muscle.A[0]/uf_d - muscle.A[0])/muscle.A[1], 1e-8]))/muscle.A[2]
    fstim_d = np.clip(fstim_d, 0.0, 40.0)
    
    _ = muscle.apply_forces(fstim_d, Lmt) # updates virtual muscle.aprime
    
    return fstim_d

control_type = int(input("Enter 1 for full PID, 2 for simple ID-PID, 3 for full ID-PID and 4 for playback: "))
'params = [pos_error, vel, integral, rel_pos, muscle, iteration, side]'
params_1 = [0]*7
params_2 = params_1.copy()
params_1[-1] = 1

if control_type == 1:
    controller = full_PID_control
elif control_type == 2:
    controller = simple_SIM_control
elif control_type == 3:
    controller = full_SIM_control
elif control_type == 4:
    fstim_array = np.array(pd.read_csv(os.path.join("misc", "MPC_fstims.csv")))
    # fstim_array = np.array(pd.read_csv(os.path.join("misc", "MPC_fstims_heart.csv")))
    fstim_array = np.vstack([fstim_array, fstim_array[-1]])
    Nt_fstim_array = np.shape(fstim_array)[0]
    playback_ratio = Nt*2/Nt_fstim_array
    def playback(params):
        'Controller for playing back stimulation frequencies over time from some other source'
        'Params = [..., iteration, side]'
        i_rel = int(params[-2]//playback_ratio)
        return fstim_array[i_rel, params[-1]]
    controller  = playback
elif control_type == 5:
    controller = ID_control
else:
    raise Exception("Invalid value entered")

params_1[4] = muscle_l
params_2[4] = muscle_r


# %%
# Open the MuJoCo viewer
i = 0
i_term = np.array([0.0, 0.0])
pos_offset = [0.4276056667, -0.4276056667]
pos_hist = np.zeros_like(target_pos_arr)
force_hist = np.zeros((Nt*2,2))
fstims = np.zeros([2,])
fstim_hist = np.zeros((Nt*2,2))
with viewer.launch_passive(model, data) as sim_viewer:
    ## Run simulation
    # while sim_viewer.is_running():
    for i in range(Nt*2):
        ## Current joint positions and velocities
        qpos = data.qpos[[0,2]] + pos_offset
        qvel = data.qvel[[0,2]]
        target_qpos = target_pos_arr[i%Nt,:]
        i_term += (target_qpos - qpos)*dt
        
        ## Control law
        angle_1 = qpos[0]
        angle_2 = qpos[1]
        params_1[0:4] = [target_qpos[0] - angle_1, qvel[0], i_term[0], angle_1]
        params_1[5] = i
        fstims[0] = controller(params_1)
        params_2[0:4] = [(angle_2 - target_qpos[1]), -qvel[1], -i_term[1], -angle_2]
        params_2[5] = i
        fstims[1] = controller(params_2)

        ## Calculate muscle forces
        theta_l = qpos[0]
        theta_r = - qpos[1]
        l_muscle_l = angle_to_muscle_length(theta_l)
        l_muscle_r = angle_to_muscle_length(theta_r)
        F_l = muscle_l.apply_forces(fstims[0], l_muscle_l)
        F_r = muscle_r.apply_forces(fstims[1], l_muscle_r)
        
        torque = np.array([muscle_force_to_moment(F_l, l_muscle_l, theta=theta_l),
                           -muscle_force_to_moment(F_r, l_muscle_r, theta=theta_r)])
        
        ## Apply torques to actuators
        data.ctrl[:] = torque
        
        ## Record current step
        pos_hist[i%Nt,:] = qpos
        force_hist[i,:] = [F_l, F_r]
        fstim_hist[i,:] = fstims

        ## Step simulation
        mujoco.mj_step(model, data)
        sim_viewer.sync()

        ## Control simulation speed (real-time)
        # time.sleep(model.opt.timestep)
        # i += 1

# %%

error = np.average(abs(target_pos_arr[:,:2] - pos_hist[:,:2]))

print(f"Average angle error: {error:.4f} rad")

plt.plot(pos_hist[:,0], pos_hist[:,1], color='C0')
plt.plot(target_pos_arr[:,0], target_pos_arr[:,1], color='orange', linestyle='dashed')

time_plt = np.linspace(0, t_final, num=np.shape(target_pos_arr)[0])
time_plt2 = np.linspace(0, t_final*2, num=np.shape(target_pos_arr)[0]*2)
plt.figure()
plt.plot(time_plt, pos_hist[:, 0], color='blue')
plt.plot(time_plt, target_pos_arr[:, 0], color='blue', linestyle='dashed')
plt.plot(time_plt, pos_hist[:, 1], color='orange')
plt.plot(time_plt, target_pos_arr[:, 1], color='orange', linestyle='dashed')
plt.ylabel('Arm Angle (rad)')
plt.xlabel('Time (sec)')
plt.title('Desired (dashed) vs Actual (solid) Arm Angles')

plt.figure()
plt.plot(time_plt, force_hist[Nt:, 0], color='blue')
plt.plot(time_plt, force_hist[Nt:, 1], color='orange')
# plt.grid(True)
plt.title('force history (Hz)')

plt.figure()
plt.plot(time_plt, fstim_hist[Nt:, 0], color='blue')
plt.plot(time_plt, fstim_hist[Nt:, 1], color='orange')
# plt.grid(True)
plt.title('fstim history (Hz)')

plt.figure()
ee_act = forward_kinematics(pos_hist.T).T
plt.plot(ee_act[:,0], ee_act[:,1], color='C0', label="Actual")
plt.plot(ee_ref[:,0], ee_ref[:,1], color='orange', linestyle='dashed', label="Desired")
plt.gca().set_aspect('equal')
plt.title(f'End Position (m), Avg Error: {error:.4f} rad')
# plt.legend()
plt.show()



# %% Save csv

# inp = input("Enter 1 to save csv: ")
# if inp == '1':
#     compressed_ctrl_hist = fstim_hist.reshape(-1, 5, 2).mean(axis=1)
#     if control_type == 1:
#         name = "PID.csv"
#     elif control_type == 2:
#         name = "simple_ID.csv"
#     elif control_type == 3:
#         name = "full_ID.csv"
#     elif control_type == 4:
#         name = "MPC.csv"
#     import os
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(base_dir, "experiment_code", "open_loop_ctrl")
#     csv_path = os.path.join(data_dir, name)
#     np.savetxt(csv_path, compressed_ctrl_hist, delimiter=",", header="ctrl1,ctrl2", comments="")


