from predictor import Predictor
from PIL import Image
import numpy as np
import cv2
from utils.basic_utils import Basic_Utils
from common import Config

pred = Predictor()

with Image.open('/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/depth_resized_image.png') as di:
            depth = np.array(di)

print(depth.shape)
with Image.open('/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/rgb_resized_image.jpg') as ri:
        rgb = np.array(ri)[:, :, :3]


#azure
#K = np.array([[976.7924194335938, 0.0, 1014.5690307617188], 
#                      [0.0, 976.2734375, 779.82177734375], 
#                      [0.0, 0.0, 1.0]], np.float32)

#ycb
K = np.array([[1066.778, 0.        , 312.9869],
                [0.      , 1067.487  , 241.3109],
                [0.      , 0.        , 1.0]], np.float32)
cam_scale = 10000

results = pred.make_prediction(rgb, depth, K, cam_scale=10000)

pred_cls_ids = results[0]
pred_pose_lst = results[1]

config = Config(ds_name="ycb")
bs_utils = Basic_Utils(config)


rgb = np.array(rgb)[:, :, :3]
rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
rgb=rgb.astype(np.uint8)
rgb = np.expand_dims(rgb, axis=0)
np_rgb = rgb[0].transpose(1, 2, 0).copy()

all_meshes_rgb = np_rgb[:, :, ::-1].copy()
for index, class_predicted in enumerate(pred_cls_ids):
        obj_id = int(class_predicted)
        pose = pred_pose_lst[index]
        mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type="ycb").copy()
        mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]

         #Prints the Meshes of the objects
        all_mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
        color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
        all_meshes_rgb = bs_utils.draw_p2ds(all_meshes_rgb, all_mesh_p2ds, r=3, color=color)
        cv2.imwrite('/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/result{}.jpg'.format(obj_id), all_meshes_rgb)
        all_meshes_rgb = np_rgb[:, :, ::-1].copy()

