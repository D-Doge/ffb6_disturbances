import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ADD import cal_adds_cuda, te, re

PATH_TO_KEYPOINTS = "datasets/ycb/ycb_kps/"
PATH_TO_OBJECT = "datasets/ycb/YCB_Video_Dataset/models/"

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


object_name = "035_power_drill"
object_pts = load_object(object_name)
rotated_points = object_pts.copy()



# Define rotation angles around each axis (in radians)
alpha = np.radians(45)  # Rotation angle around X-axis
beta = np.radians(90)   # Rotation angle around Y-axis
gamma = np.radians(180)  # Rotation angle around Z-axis

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
translation_vector = np.array([0, 0, 0]) #<------------------------------------------------------

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


object_pts = object_pts + [-0.2, 0, 0]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D points
ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2], c='b', marker='o', label='No rotatation')
ax.scatter(rotated_point_one[:, 0], rotated_point_one[:, 1], rotated_point_one[:, 2], c='#FFA500', marker='o', label='R')
ax.scatter(rotated_point_z[:, 0], rotated_point_z[:, 1], rotated_point_z[:, 2], c='r', marker='o', label='R + Z pi')
ax.scatter(rotated_point_x[:, 0], rotated_point_x[:, 1], rotated_point_x[:, 2], c='g', marker='o', label='R + X pi')
ax.scatter(rotated_point_y[:, 0], rotated_point_y[:, 1], rotated_point_y[:, 2], c='#800080', marker='o', label='R + Y pi')


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
output_file = 'output.png'
plt.savefig(output_file)

# Print a message indicating that the plot has been saved
print(f"Plot saved as '{output_file}'")

# Optionally, you can close the plot to free up memory
plt.close(fig)

