''' Class to run mujoco path following robot sim with given controller and muscles '''

import mujoco
import numpy as np

class runADAM():
    def __init__(self,
                 muscle_model, # muscle model class
                 t_loop = 15.0, #sec, time for single loop
                 num_loops = 2, # number of loops for the sim to complete
                 dt_exp = 0.01, #sec, timestep of controller/experiment code
                 model_path = "proper_mujoco_model/robot.xml", # file path to mujoco model
                 ):
        self.muscle_model = muscle_model
        self.t_loop = t_loop
        self.num_loops = num_loops
        self.dt_exp = dt_exp
        self.model_path = model_path
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.dt_sim = self.model.opt.timestep # s, Timestep of sim
        self.Nt_loop = int(self.t_loop/self.dt_sim) # Number of timesteps

        ## Get link properties from model
        self.l1 = 0.1530 #np.linalg.norm(model.body('r2').pos)
        self.l2 = 0.1157 #np.linalg.norm(model.site('EE').pos)

        ## Morphology variables
        self.delta_x = 1.853e-02
        self.delta_y = 5.375e-03
        self.r_muscle_attach = 9.572e-03

        ## Calculated morphology variables
        self.phi = np.arctan2(self.delta_y, self.delta_x)
        self.r_OC = np.sqrt(self.delta_x**2 + self.delta_y**2) # vector from fixed muscle attachment to 5bar origin

        self.L0 = .01602 # m, rest length of I3
        self.muscle_r = self.muscle_model(self.dt_sim)
        self.muscle_l = self.muscle_model(self.dt_sim)
        self.muscle_r_ideal = self.muscle_model(self.dt_sim)
        self.muscle_l_ideal = self.muscle_model(self.dt_sim)

        ## Manipulator values
        self.pos_offset = [0.4276056667, -0.4276056667] #rad, offset of arms off of straight ahead when in mujoco sim default state
        
        ## Get Target Path Values
        self.ee_ref_circ = self.circle_reference_trajectory() # End effectory target path for circle
        self.ee_ref_heart = self.heart_reference_trajectory()
        self.target_pos_circ = self.angle_trajectory(self.circle_reference_trajectory) # Arm angle target path for circle
        self.target_pos_heart = self.angle_trajectory(self.heart_reference_trajectory)
        
        ## Variable passed into controllers
        'params = [pos_error, vel, integral, rel_pos, muscle, iteration, side, circle? or heart, fstim_prev]'
        self.params_1 = [0]*9
        self.params_2 = self.params_1.copy()
        self.params_1[6] = 1
        self.params_1[4] = self.muscle_l_ideal
        self.params_2[4] = self.muscle_r_ideal
        
        ## Variables used for calibrating the lengths of the muscles
        self.force_cal = self.muscle_r.Fmto*self.muscle_r.passive(1.15)
        self.length_cal = self.muscle_r.inverse_passive(self.force_cal/self.muscle_r.Fmto)*self.muscle_r.Lmto 
        self.length_adjust = 0.0


    def set_muscles(self, new_activation, new_hill):
        ''' 
            Function to set muscles to have new fit variables, automatically resets muscles as well
            new_activation = [[A], [beta, tau, a0, g]]
            new_hill = [[C], [E], [Y], [F_mto, L_mto, L0, K_t]]
            A, C, E and Y are 1D np.arrays holding the respective fit variables
        '''
        for muscle in [self.muscle_r, self.muscle_l]:
            # Set activation fit variables
            muscle.A = new_activation[0]
            muscle.beta = new_activation[1][0]
            muscle.tau = new_activation[1][1]
            muscle.a0 = new_activation[1][2]
            muscle.g = new_activation[1][3]
            
            # # Set hill-model fit variables
            muscle.C = new_hill[0]
            muscle.E = new_hill[1]
            muscle.Y = new_hill[2]
            muscle.Fmto = new_hill[3][0]
            muscle.Lmto = new_hill[3][1]
            muscle.L0 = new_hill[3][2]
            muscle.KT = new_hill[3][3]
            
            # Recalculate simplifying variables
            muscle.passive_lmt = (muscle.C[3] + np.log(-muscle.C[0]/muscle.C[1]))/muscle.C[2]
            # muscle.slackrate = -muscle.inverse_force_velocity(0)
            
            # Reset muscle
            muscle.time = 0.0 # This will cause the muscle.lm to reset automatically when the sim starts
            muscle.aprime = 0.0 # Resets the other state variable for the activation
            
        cur_length_calibration = self.muscle_r.inverse_passive(self.force_cal/self.muscle_r.Fmto)*self.muscle_r.Lmto
        self.length_adjust = cur_length_calibration - self.length_cal
    
# %% Manip Helpers

    def angle_to_muscle_length_velocity(self, theta, dtheta):
        'Finds muscle length and velocity given specific angle of actuated arm'
        # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
        muscle_length = np.sqrt(self.delta_x**2 + self.delta_y**2 + self.r_muscle_attach**2
                                 + 2*self.delta_y*self.r_muscle_attach*np.cos(theta)
                                 - 2*self.delta_x*self.r_muscle_attach*np.sin(theta))
        muscle_velocity = -(self.delta_x*np.cos(theta) + self.delta_y*np.sin(theta))*self.r_muscle_attach*dtheta/muscle_length
        return muscle_length, muscle_velocity

    def angle_to_muscle_length(self, theta):
        'Finds muscle length given specific angle of actuated arm'
        # Uses theta as angle of the respective arm off of vertical in the direction of the muscle
        muscle_length = np.sqrt(self.delta_x**2 + self.delta_y**2 + self.r_muscle_attach**2
                                 + 2*self.delta_y*self.r_muscle_attach*np.cos(theta)
                                 - 2*self.delta_x*self.r_muscle_attach*np.sin(theta))
        return muscle_length

    def inverse_kinematics(self, p):
        'Calculates IK of 5 bar linkage for actuated arm angles giveen end effector position p=[x,y]'
        acos_term = np.arccos((p[0]**2 + p[1]**2 + self.l1**2 - self.l2**2)/(2*self.l1*np.sqrt(p[0]**2 + p[1]**2)))
        atan_term = np.arctan(p[0]/p[1])
        # theta_r = np.pi/2 - (acos_term + atan_term) # Pi terms appear there because this math was originally done for the "relative" thetas off vertical
        # theta_l = (acos_term - atan_term) + np.pi/2
        theta_r = -(acos_term + atan_term)
        theta_l = (acos_term - atan_term)
        return np.array([theta_l, theta_r])

    def forward_kinematics(self, x):
        'Calculate forward kinematics of 5 bar linkage end effector x and y positions'
        x = x + np.pi/2
        mid = np.array([self.l1/2*(np.cos(x[0]) + np.cos(x[1])), 
                         self.l1/2*(np.sin(x[0]) + np.sin(x[1]))]) # Midpoint between actuated arms
        avg_angle = (x[0] + x[1])/2
        h = np.sqrt(self.l2**2 - self.l1**2/4*(2 - 2*np.cos(x[0] - x[1]))) # Equation for dist from line between actuated arm ends to end effector
        e_perp = np.array([np.cos(avg_angle), np.sin(avg_angle)])
        endpoint = mid + h * e_perp # Two possible intersection points, choose higher "up" point
        return endpoint

    def circle_reference_trajectory(self):
        'Circle reference trajectory for end effector'
        # NOTE: Should not have arm angle less than 0.48 rad relative
        center = np.array([0.0, 0.185])  # m
        radius = 0.025 # m
        return np.array([center + radius * np.array([np.cos(2*np.pi*i), np.sin(2*np.pi*i)])
            for i in np.linspace(0, 1, num=self.Nt_loop, endpoint=False)]) # Returns an [Nt, 2] array

    def heart_reference_trajectory(self):
        'Heart shaped reference trajectory for end effector'
        center = np.array([0.0, 0.175])  # m
        scale = 0.0017  # scales to about 0.025 m width/height

        heart_points = np.array([
            center + scale * np.array([
                16 * np.sin(i)**3,
                13 * np.cos(i) - 5 * np.cos(2*i) - 2 * np.cos(3*i) - np.cos(4*i)
            ])
            for i in np.linspace(0, 2*np.pi, num=self.Nt_loop, endpoint=False)
        ])
        return heart_points # Returns an [Nt, 2] array

    def angle_trajectory(self, end_traj_func):
            'Calculates angle trajectory of a reference end effector trajectory'
            end_traj = end_traj_func()
            angle_traj = np.zeros_like(end_traj)
            for i in range(np.shape(end_traj)[0]):
                angle_traj[i,:] = self.inverse_kinematics(end_traj[i,:])
            return angle_traj # Returns an [Nt, 2] array

    def muscle_length_to_angle(self, muscle_length):
        'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
        theta = np.pi/2 + self.phi - np.arccos((self.r_muscle_attach**2 + self.r_OC**2 - muscle_length**2)/(2*self.r_muscle_attach*self.r_OC))
        return theta

    def muscle_force_to_moment(self, F, muscle_length, theta = None):
        'Converts force from muscle to moment on the respective arm'
        'MUST provide either muscle_length or theta'
        if theta is None:
            theta = self.muscle_length_to_angle(muscle_length)
        M = self.r_muscle_attach*F*(self.delta_x*np.cos(theta) + self.delta_y*np.sin(theta))/muscle_length
        return M
        

    

# %% Running Sim
    
    def run_sim(self, circle_path, controller, detailed_return=False, muscle_calibration=False):
        '''
        Function to reset and run mujoco sim of robot with a given controller, does not open a viewer
        
        Parameters
        ----------
        circle_path : Bool
            DESCRIPTION. True if circle path is desired, False if heart desired
        controller : Function
            DESCRIPTION. Function to control the manipulator
        detailed_return : Bool
            DESCRIPTION. True if full angle path history return is desired, False if just average error desired

        Returns
        -------
        avg_error : Float
            DESCRIPTION. Average error of the arm angles from their desired values for path following during the final loop
        pos_hist : Array, optional
            DESCRIPTION. Angle history of arms over full path
        force_hist : Array, optional
            DESCRIPTION. Force history of muscles over full path
        fstim_hist : Array, optional
            DESCRIPTION. Stimulation frequency history of muscles over full path
        '''
        ## Set which path to trace
        if circle_path:
            path = self.target_pos_circ
            self.params_1[7] = 1
            self.params_2[7] = 1
        else:
            path = self.target_pos_heart
            self.params_1[7] = 0
            self.params_2[7] = 0
        
        ## Reset mujoco sim
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # resets dependent variables
        
        ## Reset ideal muscles, these are fed into the SIM controllers but it is easier to manage them here
        self.muscle_r_ideal.time = 0.0
        self.muscle_r_ideal.aprime = 0.0
        self.muscle_l_ideal.time = 0.0
        self.muscle_l_ideal.aprime = 0.0

        ## Sim variable initialization
        i_term = np.array([0.0, 0.0]) # term used for integrator part of PID
        pos_hist = np.zeros((self.Nt_loop*self.num_loops,2))
        force_hist = np.zeros((self.Nt_loop*self.num_loops,2))
        fstims = np.zeros([2,])
        fstim_hist = np.zeros((self.Nt_loop*self.num_loops,2))
        
        ## Run sim
        for i in range(self.Nt_loop*self.num_loops):
            ## Current joint positions and velocities
            qpos = self.data.qpos[[0,2]] + self.pos_offset
            qvel = self.data.qvel[[0,2]]
            target_qpos = path[i%self.Nt_loop,:]
            i_term += (target_qpos - qpos)*self.dt_sim
            
            ## Control law
            angle_1 = qpos[0]
            angle_2 = qpos[1]
            self.params_1[0:4] = [target_qpos[0] - angle_1, qvel[0], i_term[0], angle_1]
            self.params_1[5] = i
            fstims[0] = controller(self.angle_to_muscle_length_velocity, self.params_1)
            self.params_2[0:4] = [(angle_2 - target_qpos[1]), -qvel[1], -i_term[1], -angle_2]
            self.params_2[5] = i
            fstims[1] = controller(self.angle_to_muscle_length_velocity, self.params_2)
            
            ## Update params for previous fstim
            self.params_1[8] = fstims[0]
            self.params_2[8] = fstims[1]
            
            ## Calculate muscle forces
            theta_l = qpos[0]
            theta_r = - qpos[1]
            
            l_muscle_l = self.angle_to_muscle_length(theta_l)
            l_muscle_r = self.angle_to_muscle_length(theta_r)
            
            if muscle_calibration:
                F_l = self.muscle_l.apply_forces(fstims[0], l_muscle_l + self.length_adjust)
                F_r = self.muscle_r.apply_forces(fstims[1], l_muscle_r + self.length_adjust)
            else:
                F_l = self.muscle_l.apply_forces(fstims[0], l_muscle_l)
                F_r = self.muscle_r.apply_forces(fstims[1], l_muscle_r)
            
            ## Step the ideal muscles forward, just like the actual muscles
            _ = self.muscle_l_ideal.apply_forces(fstims[0], l_muscle_l)
            _ = self.muscle_r_ideal.apply_forces(fstims[1], l_muscle_r)
            
            
            torque = np.array([self.muscle_force_to_moment(F_l, l_muscle_l, theta=theta_l),
                               -self.muscle_force_to_moment(F_r, l_muscle_r, theta=theta_r)])
            
            ## Apply torques to actuators
            self.data.ctrl[:] = torque
            
            ## Record current step
            pos_hist[i,:] = qpos
            force_hist[i,:] = [F_l, F_r]
            fstim_hist[i,:] = fstims

            ## Step simulation
            mujoco.mj_step(self.model, self.data)
            
        ## Calculate error
        last_loop_start = self.Nt_loop*(self.num_loops-1)
        avg_error = np.average(abs(path[:,:2] - pos_hist[last_loop_start:,:2]))
        
        ## Returns, technically this is bad coding practice, I don't care
        if detailed_return:
            return avg_error, pos_hist, force_hist, fstim_hist
        else:
            return avg_error






