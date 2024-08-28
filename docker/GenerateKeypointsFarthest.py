import open3d as o3d
import numpy as np

def farthest_point_sampling(points, num_points):
    selected_indices = []
    distance_matrix = np.ones(len(points)) * 1e10

    # Choose a random point index to start
    initial_index = np.random.randint(len(points))
    selected_indices.append(initial_index)
    furthest_point = initial_index

    for i in range(1, num_points):
        current_point = points[furthest_point]
        current_distances = np.linalg.norm(points - current_point, axis=1)
        distance_matrix = np.minimum(distance_matrix, current_distances)

        furthest_point = np.argmax(distance_matrix)
        selected_indices.append(furthest_point)

    return selected_indices

# Function to read file, filter lines, and save to a new file
def filter_lines(input_file, output_file):
    # Open the input file in read mode
    with open(input_file, 'r') as file:
        # Read all lines
        lines = file.readlines()

        # Filter lines that start with 'v ' and remove the first and second character
        filtered_lines = [line[2:] for line in lines if line.startswith('v ')]

    # Write the filtered lines to a new file
    with open(output_file, 'w') as file:
        file.writelines(filtered_lines)


def copyFile(fileIn, fileOut):
        # open both files 
    with open(fileIn,'r') as firstfile, open(fileOut,'w') as secondfile: 
    
    	# read content from first file 
    	for line in firstfile: 
        
    			# append content to second file 
    			secondfile.write(line)


object_tags = ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
              "008_pudding_box","009_gelatin_box", "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser", "024_bowl",
              "025_mug", "035_power_drill","036_wood_block", "037_scissors",
              "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]
for n_kps in [4, 16, 32, 64, 128]:
        print("Running: " + str(n_kps))
        for object_tag in object_tags:
            #Only has to be executed once per Object and saves it in the path
            ## Replace 'input.txt' with the path to your input file
            #input_file_path_raw = '../../../../Dataset/unzip/models/'+object_tag+'/textured.obj'
 
            ## Replace 'output.txt' with the desired path and name for the output file
            #output_file_path_downsampled = '../../../../Dataset/unzip/models/'+object_tag+'/obj_points_raw.xyz'
 
            ## Call the function to filter lines and save to a new file
            #filter_lines(input_file_path_raw, output_file_path_downsampled)
 
            ## Example usage
            ## Assuming you have a point cloud or set of points
            # Load the point cloud
            point_cloud = o3d.io.read_point_cloud('../../../../Dataset/unzip/models/'+object_tag+'/obj_points_raw.xyz')
            points = np.asarray(point_cloud.points)

            # Number of points to select
            num_points_to_select = n_kps

            # Apply farthest point sampling algorithm
            selected_indices = farthest_point_sampling(points, num_points_to_select)

            # Retrieve the selected points
            selected_points = points[selected_indices]

            # Create a new point cloud from the selected points
            selected_point_cloud = o3d.geometry.PointCloud()
            selected_point_cloud.points = o3d.utility.Vector3dVector(selected_points)


            o3d.io.write_point_cloud('../../../../Dataset/unzip/models/'+object_tag+'/obj_points_sampled_'+str(n_kps)+'.xyz', selected_point_cloud, write_ascii=True)

            copyFile('../../../../Dataset/unzip/models/'+object_tag+'/obj_points_sampled_'+str(n_kps)+'.xyz', '../../../../Dataset/unzip/Keypoint_frames_all/'+object_tag+'_'+str(n_kps)+'_kps.txt')
