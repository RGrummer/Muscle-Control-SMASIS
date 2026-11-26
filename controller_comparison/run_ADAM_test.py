from run_ADAM import runADAM
import os
import sys
import numpy as np
import matplotlib.pyplot as plt



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
                      )
dt = sim_manager.dt_sim

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
# kp_h = 15.0e0  # proportional gains
# kd_h = 10.0e-1  # derivative gains
# ki_h = 35.0e-1 # integral gains

## Simple ID time parameter
tau_aprime_simple = 3.0

## Full ID time parameters
tau_lm = 1.1
tau_aprime = 1.0

def full_PID_control(angle2muscle, params):
    'params = [pos_error, vel, integral]'
    fstim = np.clip(kp_f*(params[0]) + ki_f*params[2] - kd_f*params[1], 0.000001, 40.0)
    return fstim

def higher_PID_control(params):
    'Higher level PID control code, returns desired force for use with ID controller'
    'params = [pos_error, vel, integral]'
    force_d = np.clip(kp_h*params[0] + ki_h*params[2] - kd_h*params[1], 0.000001, 1.61)
    return force_d

def simple_SIM_control(angle2muscle, params):
    'Simple Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
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

def full_SIM_control(angle2muscle, params):
    'Full Hill Setpoint Inverse-Model lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
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
    
    return fstim_d

def ID_control(angle2muscle, params):
    'Inverse dynamics lower level control with PID higher level control'
    'params = [pos_error, vel, integral, rel_pos, muscle]'
    F_d = higher_PID_control(params)
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
    
    _ = muscle.apply_forces(fstim_d, Lmt) # updates virtual muscle.aprime
    
    return fstim_d


# %% Run sim

error = sim_manager.run_sim(True, full_SIM_control)
print("Error = " + str(error))




