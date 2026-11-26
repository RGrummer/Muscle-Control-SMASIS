''' Code for making manipulator design decisions '''
# All units will be in m
# This is technically a 5 bar linkage with the fixed bar having 0 length
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

## I3 value estimates based on Sukhnandan et al. (2024)
L0 = .016 # m, rest length of I3
Lmto = 0.01803 # m, length of max I3 force
Fmto = 1.61 # N, max force of I3 at Lmto

## Limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = Lmto*(1+percent_L_range) # Max I3 operation length
L_min = Lmto*(1-percent_L_range) # Min I3 operation length

## Design variables
delta_x = 1.853e-02 # Horizontal difference between fixed muscle mount point and 5bar origin
delta_y = 5.375e-03 # Vertical difference between fixed muscle mount point and 5bar origin
r_muscle_attach = 9.572e-03 # radius of arm muscle attachment point from 5bar origin
l1 = 1.530e-01 # Inner arm length aka l1 in 5bar
l2 = 1.157e-01 # Outer arm length aka l2 in 5bar
# delta_x = 1.353e-02 # Horizontal difference between fixed muscle mount point and 5bar origin
# delta_y = 5.375e-03 # Vertical difference between fixed muscle mount point and 5bar origin
# r_muscle_attach = 17.572e-03 # radius of arm muscle attachment point from 5bar origin
# l1 = 3.530e-02 # Inner arm length aka l1 in 5bar
# l2 = 3.157e-02 # Outer arm length aka l2 in 5bar

## Calculated variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

def force_length(lmt):
    'Force length relationship for I3 muscle including contractile element and series elastic element'
    B = np.array([7.44, -29.74, 37.17, -13.87])
    LT = B[0]*lmt**3 + B[1]*lmt**2 + B[2]*lmt + B[3]
    return LT

def forward_kinematics(theta_l, theta_r):
    'Calculate forward kinematics of 5 bar linkage endpoint x and y positions'
    mid_l = np.array([-l1*np.sin(theta_l), l1*np.cos(theta_l)]) # Endpoint of left l1 arm
    mid_r = np.array([l1*np.sin(theta_r), l1*np.cos(theta_r)]) # Endpoint of right l1 arm
    d = mid_r - mid_l
    midpoint = (mid_r + mid_l)/2
    d_norm = np.linalg.norm(d)

    ## Check for setup issues
    if d_norm > 2 * l2 or d_norm == 0:
        raise ValueError('Invalid, l2 too short or arm ends overlapping')

    h = np.sqrt(l2**2 - (d_norm / 2)**2) # Distance from midpoint to intersection point
    e_perp = np.array([-d[1], d[0]]) / d_norm # Unit vec perpendicular to midpoint connection lines
    endpoint = midpoint + h * e_perp # Two possible intersection points, choose higher "up" point

    return endpoint

def muscle_length_to_angle(muscle_length):
    'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
    theta = np.pi/2 + phi - np.arccos((r_muscle_attach**2 + r_OC**2 - muscle_length**2)/(2*r_muscle_attach*r_OC))
    return theta

def muscle_force_to_moment(F, muscle_length):
    'Converts force from muscle to moment on the respective arm'
    theta = muscle_length_to_angle(muscle_length)
    M = r_muscle_attach*F*(delta_x*np.cos(theta) + delta_y*np.sin(theta))/muscle_length
    return M
    
def plot_linkage(muscle_length_l, muscle_length_r, extra_points=None):
    'Plot linkage given specific muscle lengths'
    theta_l = muscle_length_to_angle(muscle_length_l)
    theta_r = muscle_length_to_angle(muscle_length_r)
    
    # Calculate important points
    px, py = forward_kinematics(theta_l, theta_r)
    endpoint = [px, py + delta_y]
    o_5bar = [0, delta_y]
    
    # Find x and y positions of middle joints between l1 and l2
    mid_joint_l = [-l1*np.sin(theta_l), l1*np.cos(theta_l) + delta_y]
    mid_joint_r = [l1*np.sin(theta_r), l1*np.cos(theta_r) + delta_y]
    
    # Setup for plot
    plot_x_wid = 2*np.max(np.abs([mid_joint_l[0], mid_joint_r[0], px, delta_x])) + l1*0.1 # last bit just for buffer
    plot_y_max = np.max([mid_joint_l[1], mid_joint_r[1], py]) + delta_y + l1*0.1
    lim_value = np.max([plot_x_wid, plot_y_max]) # ensures graph is square
    linkage_parts = [[o_5bar, mid_joint_l],
                     [o_5bar, mid_joint_r],
                     [mid_joint_l, endpoint],
                     [mid_joint_r, endpoint]]
    muscle_attach_cur_r = [-r_muscle_attach*np.sin(theta_l), r_muscle_attach*np.cos(theta_l) + delta_y]
    muscle_attach_cur_l = [r_muscle_attach*np.sin(theta_r), r_muscle_attach*np.cos(theta_r) + delta_y]
    muscles = [[[-delta_x, 0], muscle_attach_cur_r],
               [[delta_x, 0], muscle_attach_cur_l]]
    
    def plot_line(figure, p1, p2, color='blue'):
        # takes in two points and adds a line between them to the plot
        figure.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
    
    ## Make plot
    fig1, ax = plt.subplots(figsize=(8, 8))
    if extra_points is not None:
        ax.plot(extra_points[0], extra_points[1], linestyle='none', marker='.')
        extra_max = np.max(np.array([X, Y]))
        lim_value = np.max([extra_max, lim_value])
        lim_value *= 1.05
    ax.set_ylim(0-0.01, lim_value)
    ax.set_xlim(-lim_value/2, lim_value/2)
    for segment in linkage_parts:
        plot_line(ax, segment[0], segment[1], color='blue')
    for segment in muscles:
        plot_line(ax, segment[0], segment[1], color='red')
    fixed_points_x = [-delta_x, delta_x, o_5bar[0]]
    fixed_points_y = [0, 0, o_5bar[1]]
    ax.plot(fixed_points_x, fixed_points_y, marker = 'o', linestyle='none', ms=15, markerfacecolor='none', markeredgecolor='Black', markeredgewidth=3)
    ax.grid(True)
    ax.set_ylabel("(m)")
    ax.set_xlabel("(m)")
    plt.show()

plot_linkage(Lmto, Lmto)

F_Lmax = force_length(L_max/Lmto)
F_Lmin = force_length(L_min/Lmto)
M_Lmax = muscle_force_to_moment(F_Lmax, L_max)
M_Lmin = muscle_force_to_moment(F_Lmin, L_min)

print("Moment at max length:", M_Lmax)
print("Moment at min length:", M_Lmin)



## PLOT WORKSPACE VISUAL

# Calculate angles based on muscle length values
num_points = 20  # Increase for better accuracy
muscle_lengths = np.linspace(L_min, L_max, num=num_points)

X, Y = [], []
theta_vals_used = []

# ## Without Hardstops
# theta_l_vals = np.zeros_like(muscle_lengths)
# theta_r_vals = np.zeros_like(muscle_lengths)
# for i in range(len(muscle_lengths)):
#     angle = muscle_length_to_angle(muscle_lengths[i])
#     theta_l_vals[i] = angle
#     theta_r_vals[i] = angle

# theta_max = np.max(theta_l_vals)
# theta_min = np.min(theta_l_vals)
# for il in range(np.shape(theta_l_vals)[0]):
#     for ir in range(np.shape(theta_r_vals)[0]):
#         try:
#             x, y = forward_kinematics(theta_l_vals[il], theta_r_vals[ir])
#             theta_vals_used.append(theta_l_vals[il])
#             X.append(x)
#             Y.append(y)
#         except ValueError:
#             continue

## With Hardstops
theta_max = .8532
theta_min = .3812
theta_vals = np.linspace(theta_min, theta_max, num=num_points)
for il in range(np.shape(theta_vals)[0]):
    for ir in range(np.shape(theta_vals)[0]):
        try:
            x, y = forward_kinematics(theta_vals[il], theta_vals[ir])
            theta_vals_used.append(theta_vals[il])
            X.append(x)
            Y.append(y)
        except ValueError:
            continue

X = np.array(X)
Y = np.array(Y)
points = np.vstack((X, Y)).T

plot_linkage(Lmto, Lmto, extra_points=[X, Y])

import alphashape
from shapely import affinity  # for rescaling

scale_factor = 1000  # to mm
scaled_points = points * scale_factor

alpha = 0.1

shape = alphashape.alphashape(scaled_points, alpha)

if shape and shape.geom_type == "Polygon":
    # Rescale back to meters
    shape = affinity.scale(shape, xfact=1/scale_factor, yfact=1/scale_factor, origin=(0, 0))
    alpha_area = shape.area

    # Plot
    x_, y_ = shape.exterior.xy
    plt.figure()
    plt.fill(x_, y_, alpha=0.3, label="Alpha Shape")
    plt.plot(X, Y, '.', label="Points")
    plt.axis('equal')
    plt.legend()
    plt.title(f"Manipulation Area: {alpha_area * 1e6:.2f} mm²")
    plt.show()
else:
    print("Alpha shape is not a valid polygon. Type:", shape.geom_type)

# Area of the alpha shape
alpha_area = shape.area

print("Reachable workspace area (mm^2):", alpha_area*1e6)


##### PLOT MOMENTS OVER CONTRACTION #####

moments = np.zeros_like(muscle_lengths)
forces = np.zeros_like(muscle_lengths)
for i in range(len(muscle_lengths)):
    forces[i] = force_length(muscle_lengths[i]/Lmto)*Fmto
    moments[i] = muscle_force_to_moment(forces[i], muscle_lengths[i])
    
    
figm, axm = plt.subplots(figsize=(6, 6))
# axm.plot(muscle_lengths*1000, forces*100)
axm.plot(muscle_lengths*1000, moments)
axm.grid(True)
axm.set_ylim(0, )
axm.set_xlabel('Muscle Length (mm)')
axm.set_ylabel('Moment On Link 1 (Nm)')
# axm.set_ylabel('Percent of Max Muscle Force Achievable')
# axm.set_xlabel('Muscle length (mm)')
plt.show()


# %% Computer Spring forces

## Limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = Lmto*(1+percent_L_range) # Max I3 operation length
L_min = Lmto*(1-percent_L_range) # Min I3 operation length
C = np.array([-1.89, 1.70, 0.75, 0.55]) # Coefficients for parallel elastic element force-length relationship
passive_lmt = (C[3] + np.log(-C[0]/C[1]))/C[2]

def passive(lmt):
    # PE force normalized by Fmto calculation
    if lmt < passive_lmt:
        fPE = 0
    else:
        fPE = C[0] + C[1]*np.exp(C[2]*lmt-C[3])
    return fPE

min_force = Fmto*passive(L_max/Lmto)
max_force = Fmto*force_length(L_min/Lmto)
min_moment = muscle_force_to_moment(min_force, L_max)
max_moment = muscle_force_to_moment(max_force, L_min)
min_angle = muscle_length_to_angle(L_max)
max_angle = muscle_length_to_angle(L_min)

k = (max_moment - min_moment)/(max_angle - min_angle)
theta_offset = -min_moment/k + min_angle # Angle of offset off of min_rel_angle 0 position, more positive is more relaxed

print(f'k = {k} Nm/rad, theta_offset = {theta_offset} rad')

## plot min_forces, max_forces lines to show k slope derivation
min_moments = np.zeros_like(muscle_lengths)
max_moments = np.zeros_like(muscle_lengths)
arm_angles = np.zeros_like(muscle_lengths)
for i in range(len(muscle_lengths)):
    min_moments[i] = muscle_force_to_moment(Fmto*passive(muscle_lengths[i]/Lmto), muscle_lengths[i])
    max_moments[i] = muscle_force_to_moment(Fmto*force_length(muscle_lengths[i]/Lmto), muscle_lengths[i])
    arm_angles[i] = muscle_length_to_angle(muscle_lengths[i])

figk, axk = plt.subplots(figsize=(8, 6))
axk.plot(arm_angles, min_moments, label='Passive Force')
axk.plot(arm_angles, max_moments, label='Max Contract')
axk.plot([arm_angles[0], arm_angles[-1]], [max_moment, min_moment], 'o-', label='Ideal Spring')
# axk.plot([theta_offset, arm_angles[-1]], [0, min_moment], 'o--', color='C2', label='Angle Offset')
axk.legend(fontsize=16, loc="best")
axk.grid()
axk.set_xlabel('Relative Arm Angle (rad)', fontsize=16)
axk.set_ylabel('Moment On Link 1 (Nm)', fontsize=16)
plt.show()


