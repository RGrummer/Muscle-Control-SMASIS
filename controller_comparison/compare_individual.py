from run_ADAM import runADAM
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% Create simulation helper

## Import muscle force class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raw_i3_force import I3MuscleForce

## Import robot xml, there is probably a cleaner way to do this
cur_file = os.path.abspath(__file__)
subfolder = os.path.dirname(cur_file)
mainfolder = os.path.dirname(subfolder)
model_path = os.path.join(mainfolder, "proper_mujoco_model", "robot.xml")

## Create sim manager
sim_manager = runADAM(I3MuscleForce, 
                      model_path=model_path,
                      num_loops = 2,
                      )
dt = sim_manager.dt_sim



# %% Setup Muscles

### All values taken from table 2 of Sukhnandan et al.

activations = [[np.array([1.54, 7.97, 0.34, 8.40]), [0.13, 0.66, 0.74, 3.82]],
               [np.array([2.90, 6.63, 0.27, 7.11]), [0.17, 0.55, 0.57, 2.33]],
               [np.array([0.93, 7.06, 0.27, 6.83]), [0.12, 0.70, 0.76, 4.17]],
               [np.array([1.55, 7.90, 0.33, 8.33]), [0.15, 0.60, 0.74, 3.85]],
               [np.array([2.82, 6.70, 0.27, 7.17]), [0.13, 0.70, 0.66, 2.94]],
               ]

ideal_activation = [np.array([1.55, 7.90, 0.33, 8.33]), [0.15, 0.60, 0.74, 3.85]]

hill_fits = [[np.array([-1.89, 1.70, 0.75, 0.55]), np.array([1.17, 13.52, 1.60, 15.36, -10.78]), np.array([5.44, -21.15, 25.16, -8.49]), [1.61, 18.03/1000, 16.02/1000, 11.95]],
             [np.array([-0.10, 1.01, 4.11, 5.23]), np.array([0.99, 13.03, 1.37, 2.32, -5.71]), np.array([8.35, -27.35, 28.94, -8.98]), [1.62, 16.31/1000, 11.31/1000, 10.08]],
             [np.array([-0.25, 1.01, 2.33, 3.21]), np.array([0.86, 15.37, 1.45, 8.16, -9.14]), np.array([9.88, -33.87, 37.20, -12.26]), [1.76, 18.46/1000, 14.45/1000, 12.25]],
             [np.array([-0.11, 0.64, 5.80, 6.74]), np.array([0.90, 13.74, 1.37, 2.83, -6.19]), np.array([20.09, -69.39, 76.57, -26.35]), [1.91, 20.44/1000, 17.44/1000, 13.21]],
             ]

ideal_hill_fit = [np.array([-1.89, 1.70, 0.75, 0.55]), np.array([0.99, 13.03, 1.37, 2.32, -5.71]), np.array([5.44, -21.15, 25.16, -8.49]), [1.61, 18.03/1000, 16.02/1000, 11.95]]

# %% Setup controllers

## PID control parameters for full PID control
# kp_f = 14e1 # proportional gains for 2 motors
# kd_f = 17e0 # derivative gains
# ki_f = 23e0 # integral gains
kp_f = 12e1 # proportional gains for 2 motors
kd_f = 18e0 # derivative gains
ki_f = 22e0 # integral gains

## PID control parameters for higher level PID control for SIM
kp_h = 4.0e1  # proportional gains
kd_h = 15.0e-1  # derivative gains
ki_h = 15.0e-1 # integral gains

## PID control parameters for higher level PID control for ID
kp_ID = 22.0e0  # proportional gains
kd_ID = 4.0e-3  # derivative gains
ki_ID = 35.0e-1 # integral gains

## Simple ID time parameter
tau_aprime_simple = 3.0

## Full ID time parameters
tau_lm = 1.1
tau_aprime = 1.0

def full_PID(angle2muscle, params):
    'params = [pos_error, vel, integral]'
    fstim = np.clip(kp_f*(params[0]) + ki_f*params[2] - kd_f*params[1], 0.000001, 40.0)
    return fstim

def higher_PID(params, kp=kp_h, kd=kd_h, ki=ki_h):
    'Higher level PID control code, returns desired force for use with ID controller'
    'params = [pos_error, vel, integral]'
    force_d = np.clip(kp*params[0] + ki*params[2] - kd*params[1], 0.000001, 1.61)
    return force_d

def simple_SIM(angle2muscle, params):
    'Simple Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID(params)
    Lmt, Vmt = angle2muscle(params[3], params[1])
    
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

def full_SIM(angle2muscle, params):
    'Full Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID(params)
    Lmt, _ = angle2muscle(params[3], params[1])
    
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
    
    _ = muscle.apply_forces(fstim_d, Lmt) # updates virtual muscle
    
    return fstim_d

def ID(angle2muscle, params):
    'Inverse dynamics lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID(params, kp=kp_ID, kd=kd_ID, ki=ki_ID)
    Lmt, Vmt = angle2muscle(params[3], params[1])
    
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
    
    _ = muscle.apply_forces(fstim_d, Lmt) # updates virtual muscle.aprime and muscle force
    
    return fstim_d

## MPC Control
fstim_array_circle = np.array(pd.read_csv(os.path.join(mainfolder, "misc", "MPC_fstims.csv")))
fstim_array_heart = np.array(pd.read_csv(os.path.join(mainfolder, "misc", "MPC_fstims_heart.csv")))
fstim_array_circle = np.vstack([fstim_array_circle, fstim_array_circle[-1]])
fstim_array_heart = np.vstack([fstim_array_heart, fstim_array_heart[-1]])
Nt_fstim_array = np.shape(fstim_array_circle)[0]
playback_ratio = sim_manager.Nt_loop*2/Nt_fstim_array
def MPC(unused, params):
    'Controller for playing back stimulation frequencies over time from some other source'
    'Params = [..., iteration, side, circle? Or heart]'
    i_rel = int(params[5]//playback_ratio)
    if params[7] == 1:
        fstim_array = fstim_array_circle
    else:
        fstim_array = fstim_array_heart
    return fstim_array[i_rel, params[6]]


# %% Run sim

fit_pairs = [[ideal_activation, ideal_hill_fit],
             [activations[0], hill_fits[0]],
             [activations[1], hill_fits[1]],
             [activations[2], hill_fits[2]],
             [activations[3], hill_fits[3]],
             ]

controller = ID
# fits = fit_pairs[2]
circle_path = True
# sim_manager.set_muscles(ideal_activation, ideal_hill_fit)
# sim_manager.set_muscles(fits[0], fits[1])
# sim_manager.set_muscles(ideal_activation, hill_fits[3])
error, pos_hist, force_hist, fstim_hist = sim_manager.run_sim(circle_path=circle_path,
                                                              controller=controller,
                                                              detailed_return=True,
                                                              muscle_calibration=False)
print(error)

if circle_path:
    plt.figure()
    ee_act = sim_manager.forward_kinematics(pos_hist.T).T
    plt.plot(ee_act[sim_manager.Nt_loop:,0], ee_act[sim_manager.Nt_loop:,1], color='C0', label="Actual")
    plt.plot(sim_manager.ee_ref_circ[:,0], sim_manager.ee_ref_circ[:,1], color='orange', linestyle='dashed', label="Desired")
    plt.gca().set_aspect('equal')
    # plt.title(f'End Position (m), Avg Error: {error:.4f} rad')
    # plt.legend()
    plt.show()
    
    time_plt = np.linspace(0, sim_manager.t_loop, num=sim_manager.Nt_loop)
    time_plt2 = np.linspace(0, sim_manager.t_loop*2, num=np.shape(pos_hist)[0])
    double_target_circ = np.tile(sim_manager.target_pos_circ, (2, 1))
    plt.figure()
    plt.plot(time_plt2, pos_hist[:, 0], color='blue')
    plt.plot(time_plt2, double_target_circ[:, 0], color='blue', linestyle='dashed')
    plt.plot(time_plt2, pos_hist[:, 1], color='orange')
    plt.plot(time_plt2, double_target_circ[:, 1], color='orange', linestyle='dashed')
    plt.ylabel('Arm Angle (rad)')
    plt.xlabel('Time (sec)')
    plt.title('Desired (dashed) vs Actual (solid) Arm Angles')

else:
    plt.figure()
    ee_act = sim_manager.forward_kinematics(pos_hist.T).T
    plt.plot(ee_act[sim_manager.Nt_loop:,0], ee_act[sim_manager.Nt_loop:,1], color='C0', label="Actual")
    plt.plot(sim_manager.ee_ref_heart[:,0], sim_manager.ee_ref_heart[:,1], color='orange', linestyle='dashed', label="Desired")
    plt.gca().set_aspect('equal')
    # plt.title(f'End Position (m), Avg Error: {error:.4f} rad')
    # plt.legend()
    plt.show()
    
    time_plt = np.linspace(0, sim_manager.t_loop, num=sim_manager.Nt_loop)
    time_plt2 = np.linspace(0, sim_manager.t_loop*2, num=np.shape(pos_hist)[0])
    double_target_heart = np.tile(sim_manager.target_pos_heart, (2, 1))
    plt.figure()
    plt.plot(time_plt2, pos_hist[:, 0], color='blue')
    plt.plot(time_plt2, double_target_heart[:, 0], color='blue', linestyle='dashed')
    plt.plot(time_plt2, np.abs(pos_hist[:, 1]), color='orange')
    plt.plot(time_plt2, np.abs(double_target_heart[:, 1]), color='orange', linestyle='dashed')
    plt.ylabel('Absolute Arm Angle (rad)')
    plt.xlabel('Time (sec)')
    plt.title('Desired (dashed) vs Actual (solid) Arm Angles')


plt.figure()
plt.plot(time_plt2, force_hist[:, 0], color='blue')
plt.plot(time_plt2, force_hist[:, 1], color='orange')
# plt.grid(True)
plt.title('force history (Hz)')
    
plt.figure()
plt.plot(time_plt2, fstim_hist[:, 0], color='blue')
plt.plot(time_plt2, fstim_hist[:, 1], color='orange')
# plt.grid(True)
plt.title('fstim history (Hz)')
plt.show()

# for controller in [full_PID, simple_SIM, full_SIM, ID]:
#     print("New Controller")
#     for fits in fit_pairs:
#         sim_manager.set_muscles(fits[0], fits[1])
#         error = sim_manager.run_sim(circle_path=True, controller=controller)
#         print(error)



# %% Debugging ID

# F_d = higher_PID(sim_manager.params_2)
# Lmt, Vmt = sim_manager.angle_to_muscle_length_velocity(sim_manager.params_2[3], sim_manager.params_2[1])
# ## Calculate main simple hill variables
# muscle = sim_manager.params_2[4]
# lmt = (Lmt+sim_manager.params_2[9])/muscle.Lmto
# vmt = Vmt/muscle.Lmto
# FV = muscle.force_velocity(vmt)
# LT = muscle.total_force_length(lmt)
# F_cur = muscle.calc_force(lmt) # current muscle force
# fPE = muscle.passive(lmt)
# F_passive = fPE*muscle.Fmto
# lt = max(0.0, min(lmt, lmt - muscle.lm))
# fCE = muscle.KT*lt #From fSEE = fCE
# Fmt = muscle.Fmto*(fCE + fPE)
# ## Calculate stimulus amount
# f_cur = np.clip(F_cur/muscle.Fmto, 0.0, 1.0) # current muscle force
# f_d = np.clip(F_d/muscle.Fmto, 0.0, 1.0) # desired muscle force
# da_d = (f_d - f_cur)/(FV*LT*dt)
# ##
# daprime_d = da_d/muscle.g # desired change in aprime unscaled activation
# uf_d = np.max([(muscle.tau*daprime_d + muscle.beta*muscle.aprime)/(1 - muscle.aprime*(1-muscle.beta)), 0.0000001])
# fstim_d = muscle.A[3] - np.log(np.max([(muscle.A[0]/uf_d - muscle.A[0])/muscle.A[1], 1e-8]))/muscle.A[2]
# fstim_d = np.clip(fstim_d, 0.0, 40.0)









