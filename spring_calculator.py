import numpy as np


## I3 value estimates based on Sukhnandan et al. (2024)
L0 = .016 # m, rest length of I3
Lmto = L0/0.89 # m, length of max I3 force
Fmto = 1.61 # N, max force of I3 at Lmto

## Design variables
delta_x = 1.853e-02
delta_y = 5.375e-03
r_muscle_attach = 9.572e-03
l1 = 1.530e-01
l2 = 1.157e-01

## Calculated variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

## Limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = Lmto*(1+percent_L_range) # Max I3 operation length
L_min = Lmto*(1-percent_L_range) # Min I3 operation length
B = np.array([7.44, -29.74, 37.17, -13.87]) # Coefficients for force-total_muscle_length relationship
C = np.array([-1.89, 1.70, 0.75, 0.55]) # Coefficients for parallel elastic element force-length relationship
passive_lmt = (C[3] + np.log(-C[0]/C[1]))/C[2]


def force_length(lmt):
    'Force length relationship for I3 muscle including contractile element and series elastic element'
    B = np.array([7.44, -29.74, 37.17, -13.87])
    LT = B[0]*lmt**3 + B[1]*lmt**2 + B[2]*lmt + B[3]
    return LT

def passive(lmt):
    # PE force normalized by Fmto calculation
    if lmt < passive_lmt:
        fPE = 0
    else:
        fPE = C[0] + C[1]*np.exp(C[2]*lmt-C[3])
    return fPE

def muscle_length_to_angle(muscle_length):
    'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
    theta = np.pi/2 + phi - np.arccos((r_muscle_attach**2 + r_OC**2 - muscle_length**2)/(2*r_muscle_attach*r_OC))
    return theta

def muscle_force_to_moment(F, muscle_length):
    'Converts force from muscle to moment on the respective arm'
    theta = muscle_length_to_angle(muscle_length)
    M = r_muscle_attach*F*(delta_x*np.cos(theta) + delta_y*np.sin(theta))/muscle_length
    return M

min_force = passive(L_max/Lmto)
max_force = Fmto*force_length(L_min/Lmto)
min_moment = muscle_force_to_moment(min_force, L_max)
max_moment = muscle_force_to_moment(max_force, L_min)
min_angle = muscle_length_to_angle(L_max)
max_angle = muscle_length_to_angle(L_min)

k = (max_moment - min_moment)/(max_angle - min_angle)
theta_offset = -min_moment/k + min_angle # Angle of offset, more positive is more relaxed

print(f'k = {k} Nm/rad, theta_offset = {theta_offset} rad')


