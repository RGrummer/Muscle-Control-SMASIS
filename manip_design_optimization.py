''' Code for making manipulator design decisions '''
# All units will be in m
# This is technically a 5 bar linkage with the fixed bar having 0 length
import numpy as np
from scipy.integrate import dblquad
import alphashape
from shapely.geometry import Point, Polygon
from shapely import affinity  # for rescaling

## I3 value estimates based on Sukhnandan et al. (2024)
L0 = .016 # m, rest length of I3
Lmto = L0/0.89 # m, length of max I3 force
Fmto = 1.61 # N, max force of I3 at Lmto

## Limit variables
percent_L_range = 0.15 # Gives ~>=0.75*Fmto for full range
L_max = Lmto*(1+percent_L_range) # Max I3 operation length
L_min = Lmto*(1-percent_L_range) # Min I3 operation length


# ## Design variables
# delta_x = 0.01 # Horizontal difference between fixed muscle mount point and 5bar origin
# delta_y = 0.01 # Vertical difference between fixed muscle mount point and 5bar origin
# r_muscle_attach = 0.01 # radius of arm muscle attachment point from 5bar origin
# l1 = 0.03 # Inner arm length aka l1 in 5bar
# l2 = 0.04 # Outer arm length aka l2 in 5bar

# ## Calculated variables
# phi = np.arctan2(delta_y, delta_x)
# r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin


def force_length(lm):
    'Force length relationship for I3 muscle including contractile element and series elastic element'
    B = np.array([7.44, -29.74, 37.17, -13.87])
    LT = B[0]*lm**3 + B[1]*lm**2 + B[2]*lm + B[3]
    return LT

def forward_kinematics(theta_l, theta_r, l1, l2):
    'Calculate forward kinematics of 5 bar linkage endpoint x and y positions'
    mid_l = np.array([-l1*np.sin(theta_l), l1*np.cos(theta_l)]) # Endpoint of left l1 arm
    mid_r = np.array([l1*np.sin(theta_r), l1*np.cos(theta_r)]) # Endpoint of right l1 arm
    d = mid_r - mid_l
    midpoint = (mid_r + mid_l)/2
    d_norm = np.linalg.norm(d)

    ## Check for setup issues
    if d_norm > 2 * l2 or d_norm == 0:
        # print("d_norm is: ", d_norm)
        raise ValueError('Invalid geometry: l2 too short or arm ends overlapping')

    h = np.sqrt(l2**2 - (d_norm / 2)**2) # Distance from midpoint to intersection point
    e_perp = np.array([-d[1], d[0]]) / d_norm # Unit vec perpendicular to midpoint connection lines
    endpoint = midpoint + h * e_perp # Two possible intersection points, choose higher "up" point

    return endpoint

def muscle_length_to_angle(muscle_length, r_muscle_attach, phi, r_OC):
    'Converts a given muscle length to an angle of the respective arm off of vertical in the direction of the muscle'
    theta = np.pi/2 + phi - np.arccos((r_muscle_attach**2 + r_OC**2 - muscle_length**2)/(2*r_muscle_attach*r_OC))
    return theta

def muscle_force_to_moment(F, muscle_length, delta_x, delta_y, r_muscle_attach, phi, r_OC):
    'Converts force from muscle to moment on the respective arm'
    theta = muscle_length_to_angle(muscle_length, r_muscle_attach, phi, r_OC)
    M = r_muscle_attach*F*(delta_x*np.cos(theta) + delta_y*np.sin(theta))/muscle_length
    return M

def get_area(theta_max, theta_min, l1, l2):
    # def det_jac(theta_l, theta_r, delta=1e-6):
    #     'Function to calculate determinant of central diff jacobian for psuedo-integration'
    #     try:
    #         x_l1, y_l1 = forward_kinematics(theta_l + delta, theta_r, l1, l2)
    #         x_l2, y_l2 = forward_kinematics(theta_l - delta, theta_r, l1, l2)
    #         x_r1, y_r1 = forward_kinematics(theta_l, theta_r + delta, l1, l2)
    #         x_r2, y_r2 = forward_kinematics(theta_l, theta_r - delta, l1, l2)
    #     except ValueError:
    #         return 0.0  # Invalid config, contributes nothing
    
    #     if any(np.isnan([x_l1, y_l1, x_l2, y_l2, x_r1, y_r1, x_r2, y_r2])):
    #         return 0.0
    
    #     dx_dtheta_l = (x_l1 - x_l2) / (2 * delta)
    #     dy_dtheta_l = (y_l1 - y_l2) / (2 * delta)
    
    #     dx_dtheta_r = (x_r1 - x_r2) / (2 * delta)
    #     dy_dtheta_r = (y_r1 - y_r2) / (2 * delta)
    
    #     # Determinant of Jacobian
    #     return abs(dx_dtheta_l * dy_dtheta_r - dx_dtheta_r * dy_dtheta_l)
    
    # # Compute area by integrating Jacobian determinant
    # area, _ = dblquad(
    #     lambda theta_r, theta_l: det_jac(theta_l, theta_r),
    #     theta_min, theta_max,  # theta_l limits
    #     lambda theta_l: theta_min,
    #     lambda theta_l: theta_max
    # )
    
    num_points = 30  # Increase for better accuracy
    theta_l_vals = np.linspace(theta_min, theta_max, num_points)
    theta_r_vals = np.linspace(theta_min, theta_max, num_points)

    X, Y = [], []

    for il in range(np.shape(theta_l_vals)[0]):
        for ir in range(np.shape(theta_r_vals)[0]):
            try:
                x, y = forward_kinematics(theta_l_vals[il], theta_r_vals[ir], l1, l2)
                X.append(x)
                Y.append(y)
            except ValueError:
                continue

    X = np.array(X)
    Y = np.array(Y)
    points = np.vstack((X, Y)).T
    
    if np.any(np.isnan(points)):
        area = 0.0
    else:
        scale_factor = 1000  # to mm
        scaled_points = points * scale_factor
        alpha = 0.1
        
        shape = alphashape.alphashape(scaled_points, alpha)
        shape = affinity.scale(shape, xfact=1/scale_factor, yfact=1/scale_factor, origin=(0, 0))
        area = shape.area
    
    return area

def cost_function(control_vars):
    # control_vars = np.array([delta_x, delta_y, r_muscle_attach, l1&l2_avg, l1-l2])
    l1 = control_vars[3] + control_vars[4]/2
    l2 = control_vars[3] - control_vars[4]/2
    
    
    ## Directly calculated variables
    phi = np.arctan2(control_vars[1], control_vars[0])
    r_OC = np.sqrt(control_vars[0]**2 + control_vars[1]**2) # vector from fixed muscle attachment to 5bar origin
    
    ## Function calculated variables
    theta_max = muscle_length_to_angle(L_max, control_vars[2], phi, r_OC)
    theta_min = muscle_length_to_angle(L_min, control_vars[2], phi, r_OC)
    
    ## Cost variables
    # area = get_area(theta_max, theta_min, control_vars[3], control_vars[4])
    area = get_area(theta_max, theta_min, l1, l2)
    F_Lmax = force_length(L_max/Lmto)
    F_Lmin = force_length(L_min/Lmto)
    M_Lmax = muscle_force_to_moment(F_Lmax, L_max, control_vars[0], control_vars[1], control_vars[2], phi, r_OC)
    M_Lmin = muscle_force_to_moment(F_Lmin, L_min, control_vars[0], control_vars[1], control_vars[2], phi, r_OC)
    M = np.min([M_Lmax, M_Lmin])
    # cost_force = M/(control_vars[3] + control_vars[4])
    cost_force = M/np.sqrt(control_vars[3])
    # cost_force = M*1.6e-7/((control_vars[3] + control_vars[4])/2)**5
    cost_r_ma = 2000*(control_vars[2]+0.998)**(-500) # Improves ease of design
    
    ## Calculate cost
    area_cost_coeff = 5e-1
    force_cost_coeff = 3e-1
    if area == 0.0:
        cost = np.float64(0.0)
    else:
        cost = area_cost_coeff/area + force_cost_coeff/cost_force + cost_r_ma
    
    return cost, area, M


def central_finite_diff(control_vars, value_func, eps=1e-4):
    'Calculate central finite differences of control variables on given value function'
    diff_mat = np.zeros_like(control_vars)
    for i in range(len(control_vars)):
        eps_vec = np.zeros_like(control_vars)
        eps_vec[i] = eps/2
        plus = control_vars + eps_vec
        minus = control_vars - eps_vec
        plus_out, _, _ = value_func(plus)
        minus_out, _, _ = value_func(minus)
        if any([plus_out == 0.0, minus_out == 0.0]):
            diff_mat[i] = 0.0
        else:
            diff_mat[i] = (plus_out - minus_out) / eps
    return diff_mat

cost_change_abs = 0.1
cost_change = cost_change_abs
cost_change_cutoff = 1e-1
i=0

delta_x = 1.773e-02
delta_y = 5.800e-03
r_muscle_attach = 1.317e-02
l1 = 1.533e-01
l2 = 1.156e-01
# control_vars = np.array([delta_x, delta_y, r_muscle_attach, l1, l2])
control_vars = np.array([delta_x, delta_y, r_muscle_attach, (l1+l2)/2, (l1-l2)])

cur_cost, _, _ = cost_function(control_vars)
min_cost = cur_cost.copy()
min_control = control_vars.copy()

while i < 500:#cost_change_abs > cost_change_cutoff and i < 200:
    i += 1
    diff_mat = central_finite_diff(control_vars, cost_function)
    old_cost = cur_cost.copy()
    control_vars = control_vars - diff_mat*5e-9
    cur_cost, _, _ = cost_function(control_vars)
    cost_change = old_cost - cur_cost
    if cur_cost < min_cost:
        min_cost = cur_cost.copy()
        min_control = control_vars.copy()
    cost_change_abs = np.abs(cost_change)
    print(f'Iteration: {i}, Current cost: {cur_cost:.3e}, Cost diff: {cost_change:.3e}')

# %%

#delta_x, delta_y, r_muscle_attach, l1, l2
print(f'delta_x = {min_control[0]:.3e}')
print(f'delta_y = {min_control[1]:.3e}')
print(f'r_muscle_attach = {min_control[2]:.3e}')
# print(f'l1 = {min_control[3]:.3e}')
print(f'l1 = {min_control[3]+min_control[4]/2:.3e}')
# print(f'l2 = {min_control[4]:.3e}')
print(f'l2 = {min_control[3]-min_control[4]/2:.3e}')

cost, area, M = cost_function(min_control)

print(f'Area = {area*1e6} mm^2')
print(f'M = {M}')



# %% Computer Spring forces

delta_x = control_vars[0]
delta_y = control_vars[1]
r_muscle_attach = control_vars[2]
l1 = control_vars[3]
l2 = control_vars[4]

## Calculated variables
phi = np.arctan2(delta_y, delta_x)
r_OC = np.sqrt(delta_x**2 + delta_y**2) # vector from fixed muscle attachment to 5bar origin

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

min_force = passive(L_max/Lmto)
max_force = Fmto*force_length(L_min/Lmto)
min_moment = muscle_force_to_moment(min_force, L_max, control_vars[0], control_vars[1], control_vars[2], phi, r_OC)
max_moment = muscle_force_to_moment(max_force, L_min, control_vars[0], control_vars[1], control_vars[2], phi, r_OC)
min_angle = muscle_length_to_angle(L_max, control_vars[2], phi, r_OC)
max_angle = muscle_length_to_angle(L_min, control_vars[2], phi, r_OC)

k = (max_moment - min_moment)/(max_angle - min_angle)
theta_offset = -min_moment/k + min_angle # Angle of offset, more positive is more relaxed

print(f'k = {k} Nm/rad, theta_offset = {theta_offset} rad')




