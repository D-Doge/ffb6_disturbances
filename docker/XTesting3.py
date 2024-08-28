import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from ADD import cal_adds_cuda, te, re

PATH_TO_KEYPOINTS = "datasets/ycb/ycb_kps/"
PATH_TO_OBJECT = "models/"

def get_points(object, number_keypoints):
    object_path = PATH_TO_KEYPOINTS + object + "_" + str(number_keypoints) + "_kps.txt"
    kps = np.loadtxt(object_path, dtype=np.float32)
    return kps

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T

def get_random_rotation_matrix(min=-1, max=1):
    return np.random.uniform(low=min, high=max, size=(3, 4))

def load_object(object):
    path = PATH_TO_OBJECT + object + "/points.xyz"
    return np.loadtxt(path, dtype=np.float32)

def add_metric(matrix_1, matrix_2, object):
    # Generate a random 3D point v
    # But it should be the complet object
    v = load_object(object)

    rotated_point_one = np.dot(v, matrix_1[:, :3].T) + matrix_1[:, 3]
    rotated_point_two = np.dot(v, matrix_2[:, :3].T) + matrix_2[:, 3]

    distances = np.linalg.norm(rotated_point_one - rotated_point_two, axis=1)
    total_distance = np.sum(distances)
    return total_distance / len(v)

def random_fit(object, number_keypoints, add=True, add_noise=False, random_matrix=None):
    #load points
    test_points = get_points(object, number_keypoints)
    rotated_points = test_points.copy()

    #Rotate points
    if random_matrix is None:
        random_matrix = get_random_rotation_matrix()

    rotated_points = np.dot(rotated_points, random_matrix[:, :3].T) + random_matrix[:, 3]
    if(add_noise):
        mean = 0
        std_dev = 0.1
        noise = np.random.normal(mean, std_dev, size=rotated_points.shape)
        rotated_points = rotated_points + noise

    #Fit points
    fit_matrix = best_fit_transform(test_points, rotated_points)

    #Calculate error betwwen matrcies
    if(add):
        return add_metric(fit_matrix, random_matrix, object)

def calc_for_object(object, add=True, add_noise=False, reuse_matrix = False):
    results = {}
    random_matrix = None
    if reuse_matrix:
        random_matrix = get_random_rotation_matrix()

    for i in range(3, 101):
        if i in (3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 60, 70, 80, 90, 100):
            results[i] = random_fit(object, i, add, add_noise, random_matrix)

    return results

def calc_for_object_iterations(object, add=True, iterations=1000, add_noise=False, reuse_matrix = False):
    end_results = None

    for i in range(iterations):
        results = calc_for_object(object, add, add_noise, reuse_matrix)
        if end_results is None:
            end_results = results.copy()
        else:
            for key in results:
                end_results[key] += results[key]
                
        print(i)

    for key in end_results:
        end_results[key] = end_results[key] / iterations

    return end_results

def cal_adds_numpy(pred_RT, gt_RT, p3ds):
    N, _ = p3ds.shape
    
    # Calculate predicted points pd
    pd = np.dot(p3ds, pred_RT[:, :3].T) + pred_RT[:, 3]
    pd = np.expand_dims(pd, axis=1).repeat(N, axis=1)
    
    # Calculate ground truth points gt
    gt = np.dot(p3ds, gt_RT[:, :3].T) + gt_RT[:, 3]
    gt = np.expand_dims(gt, axis=0).repeat(N, axis=0)
    
    # Calculate distances between predicted and ground truth points
    dis = np.linalg.norm(pd - gt, axis=2)
    
    # Calculate minimum distances along each row (axis=1)
    mdis = np.min(dis, axis=1)
    
    # Calculate mean of minimum distances
    mean_mdis = np.mean(mdis)
    
    return mean_mdis

def cal_add_numpy(pred_RT, gt_RT, p3ds):
    pred_RT_rot = pred_RT[:, :3].T  # Transpose of the first three columns (rotation part)
    pred_RT_trans = pred_RT[:, 3]    # Last column (translation part)

    gt_RT_rot = gt_RT[:, :3].T      # Transpose of the first three columns (rotation part)
    gt_RT_trans = gt_RT[:, 3]        # Last column (translation part)

    pred_p3ds = np.dot(p3ds, pred_RT_rot) + pred_RT_trans
    gt_p3ds = np.dot(p3ds, gt_RT_rot) + gt_RT_trans

    dis = np.linalg.norm(pred_p3ds - gt_p3ds, axis=1)
    return np.mean(dis)


object_name = "035_power_drill"
object_pts = load_object(object_name)
rotated_points = object_pts.copy()

threshold_y = 0.035
#object_pts = object_pts[object_pts[:, 1] >= threshold_y]

# Define rotation angles around each axis (in radians)
alpha = np.radians(120)  # Rotation angle around X-axis
beta = np.radians(90)   # Rotation angle around Y-axis
gamma = np.radians(280)  # Rotation angle around Z-axis

# Rotation matrix around X-axis (Rx)
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha)],
    [0, np.sin(alpha), np.cos(alpha)]
])

# Rotation matrix around Y-axis (Ry)
Ry = np.array([
    [np.cos(beta), 0, np.sin(beta)],
    [0, 1, 0],
    [-np.sin(beta), 0, np.cos(beta)]
])

# Rotation matrix around Z-axis (Rz)
Rz = np.array([
    [np.cos(gamma), -np.sin(gamma), 0],
    [np.sin(gamma), np.cos(gamma), 0],
    [0, 0, 1]
])

# Combined rotation matrix (perform rotations in sequence: ZYX order)
rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
# Translation vector (0.1 meters along the x-axis)
translation_vector = np.array([0, 0.02, 0]) #<------------------------------------------------------

# Create the combined transformation matrix (rotation followed by translation)
# Stack the rotation_matrix with the translation_vector to form a (3, 4) matrix
rotation_translation_matrix = np.column_stack((rotation_matrix, translation_vector))

rotated_point_one = np.dot(object_pts, rotation_translation_matrix[:, :3].T) + rotation_translation_matrix[:, 3]

gamma = np.radians(180)
Rz = np.array([
    [np.cos(gamma), -np.sin(gamma), 0],
    [np.sin(gamma), np.cos(gamma), 0],
    [0, 0, 1]
])

rotated_point_z = np.dot(object_pts, Rz @ rotation_translation_matrix[:, :3].T) + rotation_translation_matrix[:, 3] + [0.2, 0, 0]

alpha = np.radians(180)
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha)],
    [0, np.sin(alpha), np.cos(alpha)]
])

rotated_point_x = np.dot(object_pts, Rx @ rotation_translation_matrix[:, :3].T) + rotation_translation_matrix[:, 3] + [0.4, 0, 0]

beta = np.radians(180)
Ry = np.array([
    [np.cos(beta), 0, np.sin(beta)],
    [0, 1, 0],
    [-np.sin(beta), 0, np.cos(beta)]
])
rotated_point_y = np.dot(object_pts, Ry @ rotation_translation_matrix[:, :3].T) + rotation_translation_matrix[:, 3] + [0.6, 0, 0]

def rotation_matrix_from_axis_angle(u, theta):
    """
    Compute the rotation matrix for rotating around a given axis by a specified angle.
    
    Parameters:
        u (numpy.ndarray): A 3D vector representing the rotation axis (unit vector).
        theta (float): The rotation angle in radians.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    # Normalize the axis vector (in case it's not already normalized)
    u = u / np.linalg.norm(u)
    
    # Components of the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ux, uy, uz = u
    
    # Compute the rotation matrix
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux*uy*(1 - cos_theta) - uz*sin_theta, ux*uz*(1 - cos_theta) + uy*sin_theta],
        [uy*ux*(1 - cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy*uz*(1 - cos_theta) - ux*sin_theta],
        [uz*ux*(1 - cos_theta) - uy*sin_theta, uz*uy*(1 - cos_theta) + ux*sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    
    return R

new_x_axis = np.array([1,0,0]) @ rotation_translation_matrix[:, :3].T
print(new_x_axis)
Rx = rotation_matrix_from_axis_angle(new_x_axis, np.radians(180))
rotated_point_x = rotated_point_one @ Rx + rotation_translation_matrix[:, 3] + [0.2, 0, 0]


object_pts = object_pts + [0, 0, 0.2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D points
ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2], c='b', marker='o', label='No rotatation')
ax.scatter(rotated_point_one[:, 0], rotated_point_one[:, 1], rotated_point_one[:, 2], c='#FFA500', marker='o', label='Translated by 0.2')
#ax.scatter(rotated_point_z[:, 0], rotated_point_z[:, 1], rotated_point_z[:, 2], c='r', marker='o', label='R + Z pi')
ax.scatter(rotated_point_x[:, 0], rotated_point_x[:, 1], rotated_point_x[:, 2], c='g', marker='o', label='R + X pi')
#ax.scatter(rotated_point_y[:, 0], rotated_point_y[:, 1], rotated_point_y[:, 2], c='#800080', marker='o', label='R + Y pi')


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud Visualization')

# Display legend
ax.legend()
# Set fixed limits for the axes (adjust these limits based on your data range)
limit = [-0.5, 0.5]
ax.set_xlim(limit)  # Set limits for the X-axis
ax.set_ylim(limit)  # Set limits for the Y-axis
ax.set_zlim(limit)  # Set limits for the Z-axis

# Save the plot to a file (replace 'output.png' with your desired file path and format)
output_file = 'output2.png'
plt.savefig(output_file)

# Print a message indicating that the plot has been saved
print(f"Plot saved as '{output_file}'")

# Optionally, you can close the plot to free up memory
plt.close(fig)


def rotation_matrix_from_axis_angle(u, theta):
    """
    Compute the rotation matrix for rotating around a given axis by a specified angle.
    
    Parameters:
        u (numpy.ndarray): A 3D vector representing the rotation axis (unit vector).
        theta (float): The rotation angle in radians.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    # Normalize the axis vector (in case it's not already normalized)
    u = u / np.linalg.norm(u)
    
    # Components of the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ux, uy, uz = u
    
    # Compute the rotation matrix
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux*uy*(1 - cos_theta) - uz*sin_theta, ux*uz*(1 - cos_theta) + uy*sin_theta],
        [uy*ux*(1 - cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy*uz*(1 - cos_theta) - ux*sin_theta],
        [uz*ux*(1 - cos_theta) - uy*sin_theta, uz*uy*(1 - cos_theta) + ux*sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    
    return R

# Example usage:
#[1,0,0] x
#[0,1,0] y
#[0,0,1] z
axis_vector = np.array([0, 0, 1])  # Example axis vector (not necessarily unit length)
angle_radians = np.pi          # Example rotation angle (60 degrees)

# Normalize the axis vector
axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)

# Compute the rotation matrix
rotation_matrix = rotation_matrix_from_axis_angle(axis_vector_normalized, angle_radians)

print("Rotation Matrix:")
print(rotation_matrix)


import math
def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose given by a 3x3 matrix.
    :param R_gt: Rotational element of the ground truth pose given by a 3x3 matrix.
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    return error

import itertools
def combine_lists(lists_dict):
    # Extract lists from the dictionary values
    lists = list(lists_dict.values())
    
    # Use itertools.product to generate all combinations
    for combo in itertools.product(*lists):
        yield combo

from scipy.optimize import minimize_scalar

def rotation_error_metric(R_gt, R_est, symmertic_dict=None, rotation_invarant_list=None):
    """
    Get the min rotation error for every correct pose

    :R_gt: 3x3 numpy array that represents the ground truth rotation matrix
    :R_est: 3x3 numpy array that represents the estimated rotation matrix
    :symmertic_dict: A dic which has tuples as keys, which represent a vector/axis on which the object has correct poses. For each key we have a list of correct angles on this axis. 0 deg must be included
    :rotation_invarant_list: list with vectors/axis on which the rotation does not matter. If it is longer than 1 the rotation does not matter.
    """
    #Sym dict contains vecors (rotation axis) as keys and per key a list of correct rotations
    #Example symmertic_dict[ (1,0,0) ] = [0, 90, 180, 270] means these 4 rotations around the x axis are correct
    #We can than simply apply all correct rotaions and calculate the rotation error and than take the min rotation error.
    #Maybe use key as bytes https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary


    #the rotation_invarant_list should only contain one vector, when it has more than 1 entry it is a ball and the rotation error is 0.
    #We should apply the rotation around this vector last and solve it as a optimisation problem.

    if symmertic_dict is None and rotation_invarant_list is None:
        return re(R_gt, R_est)
    
    if symmertic_dict is None:
        #Only rotation invariance
        if len(rotation_invarant_list >= 2):
            return 0
        
        rotation_axis = rotation_invarant_list[0] @ R_gt
        def function_to_optimize(rotation_agnle):
            new_R_gt = rotation_matrix_from_axis_angle(rotation_axis, rotation_agnle) @ R_gt
            return re(new_R_gt, R_est)

        optimal_rotation_angle = minimize_scalar(function_to_optimize).x

        return function_to_optimize(optimal_rotation_angle)
    
    if rotation_invarant_list is None:
        #Only sym
        error_list = []
        combos = combine_lists(symmertic_dict) # combos list of tuple with every rotation combo
        vector_list = list(symmertic_dict.keys()) # list of all rotation vecotrs
        for combo in combos: #take one rotation combo
            new_R_gt = R_gt 
            for i in range(len(vector_list)): #iterate over every rotaion axis
                new_rotation_axis = np.array(vector_list[i]) @ new_R_gt #apply current rotation to the rotation axis (algin to object cords)
                new_R_gt = rotation_matrix_from_axis_angle(new_rotation_axis, combo[i]) @ new_R_gt #add the rotation to the current rotation matrix
            error_list.append(re(new_R_gt, R_est)) #Calc error for the new rotation matrix
        
        return min(error_list) #return min error
    
    

    #both

    error_list = []
    combos = combine_lists(symmertic_dict) # combos list of tuple with every rotation combo
    vector_list = list(symmertic_dict.keys()) # list of all rotation vecotrs
    for combo in combos: #take one rotation combo
        new_R_gt = R_gt 
        for i in range(len(vector_list)): #iterate over every rotaion axis
            new_rotation_axis = np.array(vector_list[i]) @ new_R_gt #apply current rotation to the rotation axis (algin to object cords)
            new_R_gt = rotation_matrix_from_axis_angle(new_rotation_axis, combo[i]) @ new_R_gt #add the rotation to the current rotation matrix

        #After applying rotations, find optimal rotation on inverant axis
        rotation_axis = rotation_invarant_list[0] @ new_R_gt
        def function_to_optimize(rotation_agnle):
            new_R_gt = rotation_matrix_from_axis_angle(rotation_axis, rotation_agnle) @ new_R_gt
            return re(new_R_gt, R_est)

        optimal_rotation_angle = minimize_scalar(function_to_optimize).x

        new_R_gt = rotation_matrix_from_axis_angle(rotation_axis, optimal_rotation_angle) @ new_R_gt
        error_list.append(re(new_R_gt, R_est)) #Calc error for the new rotation matrix
    
    return min(error_list) #return min error
    
def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose given by a 3x1 vector.
    :param t_gt: Translation element of the ground truth pose given by a 3x1 vector.
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

def get_rotations_for_cls(cls_id):
    return (None, None)

def error_metric(R_gt, t_gt, R_est, t_est, cls_id):
    if R_gt is None: #Case False dection
        return math.inf
    if R_est is None: #Case Object was not dected
        return math.inf

    symmertic_dict, rotation_invarant_list = get_rotations_for_cls(cls_id)
    rotation_error = rotation_error_metric(R_gt, R_est, symmertic_dict, rotation_invarant_list) / math.pi # Scale rotation error to be 0 <= rotation_error <= 1, so it has the same weight as the translation error
    translation_error = te(t_est, t_gt)
    translation_error = min(translation_error/3.5, 1) #normalize the translation error it should be 1 at 3.5m, since it is the max range of asus xtion depth sensor, larger error do not make much sence

    return rotation_error + translation_error

def dd_score(erro_list):
    sum = 0
    for error in erro_list:
        if error == math.inf: #Case wrong dection add nothing to the sum
            continue

        sum = sum + (1/(1+error)) #add 1 if pereft pose, add 0.33 in worst case

    return sum / len(erro_list) #Normilze between 0 and 1
    

test_dict = {}
test_dict[(1,0,0)] = [0, 90, 180, 270]
test_dict[(0,1,0)] = [0, 90, 180, 270]

for axis in test_dict:
    print("----------")
    print(axis)
    print(test_dict[axis])
