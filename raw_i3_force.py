import numpy as np

class I3MuscleForce():
    '''
        Uses a hill type muscle model to output a raw force value
        CE = Contractile Element, PE = Passive Element, SEE = Series Elastic Element
        "lt" refers to  SEE length and "lm" refers to CE length
        If you change this code to model a different muscle, make sure to retune the FV clipping on line 74
    '''
    
    def __init__(self, dt, L0=0.01602):
        # L0 = approx muscle rest lenght in m, circumference of buccal mass/2
        # fstim = frequency of electrical stimulation, callable class used so frequency
            #can be updated over time
        # dt = timestep of simulation

        self.L0 = L0 # unit does not matter so long as it is the same as Lmt in apply_forces
        self.dt = dt
        self.time = 0.0
        
        self.lt0 = 0.0 # Always initialize at 0 given no initial CE activation
        
        # Muscle model fit constants from Sukhnandan et al.
        self.A = np.array([1.55, 7.90, 0.33, 8.33])
        self.B = np.array([7.44, -29.74, 37.17, -13.87]) # Unused by default Hill model
        self.C = np.array([-1.89, 1.70, 0.75, 0.55])
        self.D = np.array([-0.77, 0.07, 0.35, 0.07]) # Unused by default Hill model
        self.E = np.array([0.99, 13.03, 1.37, 2.32, -5.71])
        self.Y = np.array([5.44, -21.15, 25.16, -8.49])
        
        self.beta = 0.15
        self.tau = 0.60
        self.a0 = 0.74
        self.g = 3.85
        
        self.KT = 11.95
        self.passive_lmt = self.inverse_passive(0)
        self.Lmto = L0/0.8885191347753743 # Relative to natural muscle length, maybe unnecessary
        self.Fmto = 1.61
        self.lm = L0/self.Lmto # Preserved across time steps, this value is immediately overwritten at the start of the sim
        
        self.aprime = 0.0 # activation prior to scaling and thresholding
        self.slackrate = 2.312#-self.inverse_force_velocity(0) # Negative IFV gives positive slackrate since want lt->0 thus lm->lmt
            #to provide a reasonable rate for tendon return after muscle relaxation
    
    def calc_force(self, lmt):
        # Calculate muscle force
        fPE = self.passive(lmt)
        lt = max(0.0, min(lmt, lmt - self.lm))
        fCE = self.KT*lt #From fSEE = fCE
        Fmt = self.Fmto*(fCE + fPE)
        
        return Fmt
    
    def apply_forces(self, fstim, Lmt):
        # Calculate normalized full body length
        lmt = Lmt/self.Lmto
        
        # Update length of tendon portion
        if self.time == 0:
            self.lm = lmt - self.lt0
            lt = self.lt0
        else:
            lt = max(0.0, min(lmt, lmt - self.lm)) # Extra safe to make sure there are no funky numbers
        
        # Calculate forces relevant to total
        fPE = self.passive(lmt)
        fCE = self.KT*lt #From fSEE = fCE
        
        # Back calculate muscle velocity "dlm" from fCE
        a = self.activation(fstim)
        LT = self.force_length(self.lm)
        if a > 0.0:
            FV = min(fCE/(a*LT), 10.0) # Caps out FV to stop sim from breaking at a << 1, rough limit to make numbers better
            dlm = min(self.inverse_force_velocity(FV), 0.1) # Actual tuning variable for mitigating spikes is the 0.1 here
        elif a == 0.0 and self.lm > 0.0:
            dlm = self.slackrate
        elif a == 0.0 and self.lm <= 0.0:
            dlm = 0.0
        else:
            raise Exception("Activation < 0")
        
        # Calculate new tendon length "lt"
        lt = max(0.0, min(lmt, lmt - (self.lm + dlm*self.dt)))
        self.lm = lmt - lt
        
        # Total force
        Fmt = self.Fmto*(fCE + fPE)
        self.time += self.dt
        
        return Fmt
    
    'Fit based functions, put here for organization:'
    
    def activation(self, fstim):
        # Calculate muscle activation amount
        # uf = FCE/FCEmax = adjust contractile force for different frequency input
        # Note, the activation will slowly rise on its own even when fstim=0
        uf = self.A[0]/(self.A[0] + 
                        self.A[1]*np.exp(-self.A[2]*(fstim - self.A[3])))
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

    def inverse_passive(self, fPE):
        # Inverts passive equation, finds muscle length to achieve certain force, unused in model
        lmt = (self.C[3] + np.log((fPE - self.C[0])/self.C[1]))/self.C[2]
        return lmt

    def force_length(self, lm):
        # CE force-length relationship
        LT = self.Y[0]*lm**3 + self.Y[1]*lm**2 + self.Y[2]*lm + self.Y[3]
        return LT
    
    def total_force_length(self, lmt):
        # Force length relationship for whole muscle, unused in full hill model but used for simple hill model
        LT = self.B[0]*lmt**3 + self.B[1]*lmt**2 + self.B[2]*lmt + self.B[3]
        return LT
    
    def force_velocity(self, dlm):
        # CE force-velocity relationship, unused in this model
        if dlm < 0:
            FV = 1 + self.D[0]/(1 - self.D[1]/dlm)
        elif dlm > 0:
            FV = 1 + self.D[2]/(1 + self.D[3]/dlm)
        else:
            FV = 1.0
        return FV
    
    def inverse_force_velocity(self, FV):
        # CE inverse force-velocity relationship, given effect of velocity on
        # the total fCE, outputs the velocity that would cause that
        dlm = self.E[0]*np.exp(self.E[1]*(FV - self.E[2])) - self.E[3]*np.exp(self.E[4]*FV)
        return dlm
            
            

