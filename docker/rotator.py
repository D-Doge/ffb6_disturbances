import numpy as np

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


#print(random_fit("002_master_chef_can", 8, add_noise=True))
#print(calc_for_object_iterations("002_master_chef_can"))
print(calc_for_object_iterations("035_power_drill", reuse_matrix=True, add_noise=True))
