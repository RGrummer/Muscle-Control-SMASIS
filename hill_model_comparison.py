import numpy as np
import math
from matplotlib import pyplot as plt


# %% EMG Data Processing          

dt = 1e-4
final_time = 20.0

# Import and Process EMG Data
I3_EMG_data = np.genfromtxt('I3_EMG_freq2length.csv', delimiter=',') #values are [timedata, freqdata, lengthdata, expforcedata, simforcedata]
    # lengthdata is given at % of normalized length
total_steps = int(final_time / dt)
I3_EMG_used = np.zeros((total_steps, 5))

for n in range(total_steps):
    t_EMG_ind = np.abs(I3_EMG_data[:,0] - dt*n).argmin() # Finds index of current time in EMG data
    I3_EMG_used[n] = I3_EMG_data[t_EMG_ind,:]
    print(n, end='\r')
print("EMG Data Processed")


# %% I3 Forcing Function
class Full_I3():
    # Uses a hill type muscle model,
    # CE = Contractile Element, PE = Passive Element, SEE = Series Elastic Element
    
    def __init__(self, L0, fstim, dt, lengths):
        # L0=approx muscle rest lenght, circumference of buccal mass/2
        # fstim = frequency of electrical stimulation
        # dt = timestep of simulation

        self.L0 = L0
        self.dt = dt
        self.fstim = fstim
        
        self.lt0 = 0.0 # Always initialize at 0 given no initial CE activation
        
        # Muscle model fit constants from Sukhnandan et al.
        self.A = np.array([1.55, 7.90, 0.33, 8.33])
        self.B = np.array([7.44, -29.74, 37.17, -13.87])
        self.C = np.array([-1.89, 1.70, 0.75, 0.55])
        self.D = np.array([-0.77, 0.07, 0.35, 0.07])
        self.E = np.array([0.8613, 15.3656, 1.4487, 8.1555, -9.1426])
        self.Y = np.array([5.44, -21.15, 25.16, -8.49])
        
        self.beta = 0.15
        self.tau = 0.60
        self.a0 = 0.74
        self.g = 3.85
        
        self.KT = 12.018
        self.Lmto = L0/0.8643 #Relative to natural muscle length, maybe unnecessary
        self.Fmto = 1.61
        
        self.passive_lmt = (self.C[3] + np.log(-self.C[0]/self.C[1]))/self.C[2]
        print(self.passive_lmt)
        
        self.lm = L0/self.Lmto # Preserved across time steps
        self.aprime = 0 # activation prior to scaling and thresholding
        # self.ft = final_time
        # self.debug_history = [] # Tracks any desired variable for debugging
        self.test = 0
        self.lengths = lengths
        
        
    def apply_forces(self, time: np.float64 = 0.0):

        # Calculate normalized full body length
        lmt = self.lengths[int(time/self.dt)]
        
        if time < self.dt:
            self.lm = lmt - self.lt0
            lt = self.lt0
        else:
            lt = max(0.0, min(lmt, lmt - self.lm))
        
        # Calculate forces relevant to total
        fPE = self.passive(lmt)
        fCE = self.KT*lt #From fSEE = fCE
        
        # Back calculate muscle velocity "dlm" from fCE
        a = self.activation(time)
        LT = self.force_length(self.lm, self.Y)
        if a > 0.0:
            FV = min(fCE/(a*LT), 1.3) # Caps out FV to stop sim from breaking at a << 1
            dlm = self.inverse_force_velocity(FV, self.E)
        elif a == 0.0 and lt > 0.0:
            FV = 0
            dlm = -self.inverse_force_velocity(0, self.E) 
                # Negative in order to provide a rate for tendon slack return
        elif a == 0.0 and lt <= 0.0:
            FV = 0
            dlm = 0.0
        else:
            raise Exception("Activation < 0")
        
        # Calculate new tendon length "lt"
        lt = max(0.0, min(lmt, lmt - (self.lm + dlm*self.dt)))
        self.lm = lmt - lt
        
        # Total force
        Fmt = self.Fmto*(fCE + fPE)
        
        return Fmt
       
    # Fit based functions, put here for organization:
    
    def activation(self, time):
        # Calculate muscle activation amount
        # uf = FCE/FCEmax = adjust contractile force for different frequency input
        uf = self.A[0]/(self.A[0] + 
                        self.A[1]*np.exp(-self.A[2]*(self.fstim[int(time/dt)] - self.A[3])))
        daprime = 1/self.tau*(uf - (self.beta + (1-self.beta)*uf)*self.aprime)
        self.aprime = self.aprime + self.dt*daprime
        a = max(0.0, min(1.0, self.g*(self.aprime - self.a0)))
        return a
        
    def passive(self, lmt):
        # PE force normalized by Fmto calculation
        if lmt < self.passive_lmt:
            fPE = 0
        else:
            fPE = self.C[0] + self.C[1]*np.exp(self.C[2]*lmt-self.C[3])
        return fPE
    
    def force_length(self, lm, Y):
        # CE force-length relationship
        LT = Y[0]*lm**3 + Y[1]*lm**2 + Y[2]*lm + Y[3]
        return LT
    
    def force_velocity(self, dlm, D):
        # CE force-velocity relationship, unused in this model
        if dlm <= 0:
            FV = 1 + D[0]/(1 - D[1]/dlm)
        else:
            FV = 1 + D[2]/(1 + D[3]/dlm)
        return FV
    
    def inverse_force_velocity(self, FV, E):
        # CE inverse force-velocity relationship, given effect of velocity on
        # the total fCE, outputs the velocity that would cause that
        dlm = E[0]*math.exp(E[1]*(FV - E[2])) - E[3]*math.exp(E[4]*FV)
        return dlm



class Simple_I3():
    # Uses a hill type muscle model,
    # CE = Contractile Element, PE = Passive Element, SEE = Series Elastic Element
    
    def __init__(self, L0, fstim, dt, lengths, velocities, final_time):
        # L0=approx muscle rest lenght, circumference of buccal mass/2
        # fstim = frequency of electrical stimulation
        # dt = timestep of simulation

        self.L0 = L0
        self.dt = dt
        self.fstim = fstim
        
        self.lt0 = 0.0 # Always initialize at 0 given no initial CE activation
        
        # Muscle model fit constants from Sukhnandan et al.
        self.A = np.array([1.55, 7.90, 0.33, 8.33])
        self.B = np.array([7.44, -29.74, 37.17, -13.87])
        self.C = np.array([-1.89, 1.70, 0.75, 0.55])
        self.D = np.array([-0.77, 0.07, 0.35, 0.07])
        self.E = np.array([0.8613, 15.3656, 1.4487, 8.1555, -9.1426])
        self.Y = np.array([5.44, -21.15, 25.16, -8.49])
        
        self.beta = 0.15
        self.tau = 0.60
        self.a0 = 0.74
        self.g = 3.85
        
        self.KT = 12.018
        self.Lmto = L0/0.8643 #Relative to natural muscle length, maybe unnecessary
        self.Fmto = 1.61
        
        self.passive_lmt = (self.C[3] + np.log(-self.C[0]/self.C[1]))/self.C[2]
        print(self.passive_lmt)
        
        self.lm = L0/self.Lmto # Preserved across time steps
        self.aprime = 0 # activation prior to scaling and thresholding
        self.lengths = lengths
        self.velocities = velocities
        
        self.ft = final_time
        self.debug_history = [] # Tracks any desired variable for debugging
        self.test = 0
        
        
    def apply_forces(self, time: np.float64 = 0.0):

        # Calculate normalized full body length
        lmt = self.lengths[int(time/self.dt)]
        dlmt = self.velocities[int(time/self.dt)]
        
        # Calculate forces relevant to total
        fPE = self.passive(lmt)
        
        # Back calculate muscle velocity "dlm" from fCE
        a = self.activation(time)
        LT = self.total_force_length(lmt)
        FV = self.force_velocity(dlmt, self.D)
        
        # self.tracker = FV
        # self.debug_history.append(self.tracker)
        # # Save debug_history to a CSV file at last time step to use for data comparison
        # # Another option for doing this would be to capture print outputs into 
        #     #a variable using io.StringIO
        # if time >= self.ft - 1.1*dt:
        #     np.savetxt("EMG_debug_history.csv", self.debug_history, delimiter=",")
        
        # Total force
        Fmt = self.Fmto*(a*LT*FV + fPE)
        return Fmt
       
    # Fit based functions, put here for organization:
    
    def activation(self, time):
        # Calculate muscle activation amount
        # uf = FCE/FCEmax = adjust contractile force for different frequency input
        uf = self.A[0]/(self.A[0] + 
                        self.A[1]*np.exp(-self.A[2]*(self.fstim[int(time/dt)] - self.A[3])))
        daprime = 1/self.tau*(uf - (self.beta + (1-self.beta)*uf)*self.aprime)
        self.aprime = self.aprime + self.dt*daprime
        a = max(0.0, min(1.0, self.g*(self.aprime - self.a0)))
        return a
        
    def passive(self, lmt):
        # PE force normalized by Fmto calculation
        if lmt < self.passive_lmt:
            fPE = 0
        else:
            fPE = self.C[0] + self.C[1]*np.exp(self.C[2]*lmt-self.C[3])
        return fPE
    
    def total_force_length(self, lmt):
        # Force length relationship for whole muscle
        LT = self.B[0]*lmt**3 + self.B[1]*lmt**2 + self.B[2]*lmt + self.B[3]
        return LT
    
    def force_velocity(self, dlm, D):
        # CE force-velocity relationship
        if dlm < 0:
            FV = 1 + self.D[0]/(1 - self.D[1]/dlm)
        elif dlm > 0:
            FV = 1 + self.D[2]/(1 + self.D[3]/dlm)
        else:
            FV = 1.0
        return FV
    


# %% Environment Setup

lengths = I3_EMG_used[:, 2]
fstim = I3_EMG_used[:, 1]*49/max(I3_EMG_used[:,1])
velocities = np.concatenate(([0], lengths[1:] - lengths[:-1]))/dt


full = Full_I3(L0=1, fstim=fstim, dt=dt, lengths=lengths)
simple = Simple_I3(L0=1, fstim=fstim, dt=dt, lengths=lengths, velocities=velocities, final_time=final_time)

total_steps = int(final_time / dt)
print("Total steps", total_steps)

end_forces_f = np.zeros((np.shape(I3_EMG_used)[0],))
end_forces_s = np.zeros((np.shape(I3_EMG_used)[0],))

for n in range(total_steps):
    end_forces_f[n] = full.apply_forces(I3_EMG_used[n,0])
    end_forces_s[n] = simple.apply_forces(I3_EMG_used[n,0])
    pass


# %% Plotting Data


figForce = plt.figure(figsize=(10, 6))
bx = figForce.add_subplot(111)

bx.plot(I3_EMG_used[:,0]+1, end_forces_f/1.24, label='Python Model')
bx.plot(I3_EMG_used[:,0]+1, end_forces_s/1.24, label='Python Simplified')
bx.plot(I3_EMG_used[:,0], I3_EMG_used[:,3], color='red', ls='--', label='EMG Data')
bx.plot(I3_EMG_used[:,0]+1, I3_EMG_used[:,4], color='orange', ls='--', label='Simulink Model')
bx.set_ylabel("Normalized Muscle Force", fontsize=20)
bx.set_xlabel("Time [s]", fontsize=20)
bx.legend(fontsize=18, loc="upper right")
bx.tick_params(axis='both', labelsize=16)

plt.show()
plt.close(plt.gcf())


fig, cx = plt.subplots(2, 1, figsize=(10, 7))

cx[0].plot(I3_EMG_used[:,0], fstim)
cx[0].set_ylabel("Stimulation Frequency (Hz)", fontsize = 14)
cx[1].plot(I3_EMG_used[:,0], I3_EMG_used[:,2])
cx[1].set_ylabel("Normalized Muscle Length", fontsize=14)
cx[1].set_xlabel("Time (s)", fontsize = 16)
cx[0].tick_params(axis='both', labelsize=12)
cx[1].tick_params(axis='both', labelsize=12)
plt.show()
plt.close(plt.gcf())


# DebugData = np.genfromtxt('EMG_debug_history.csv', delimiter=',')

# figDebug = plt.figure(figsize=(10, 8))
# dx = figDebug.add_subplot(111)

# dx.plot(I3_EMG_used[:int(np.shape(DebugData)[0]),0], DebugData)
# # dx.set_ylim(0.75, 0.9)
# plt.grid()
# plt.show()
# plt.close(plt.gcf())







