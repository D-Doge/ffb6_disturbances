from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
import datasets.ycb.ycb_dataset_REAL_IMAGE as dataset_desc
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config(ds_name="ycb")

bs_utils = Basic_Utils(config)


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        #print(end_points['pred_rgbd_segs'].shape)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
            pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
            end_points['pred_kp_ofs'][0], True, config.n_objects, True,
            None, None
        )
        print("Found the following classes: ")
        print(pred_cls_ids)
        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        predictions_rgb = np_rgb[:, :, ::-1].copy()
        meshes_rgb = np_rgb[:, :, ::-1].copy()
        all_predictions_rgb = np_rgb[:, :, ::-1].copy()
        all_meshes_rgb = np_rgb[:, :, ::-1].copy()
        
        for index, class_predicted in enumerate(pred_cls_ids):
            obj_id = int(class_predicted)
            pose = pred_pose_lst[index]
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type="ycb").copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            K = config.intrinsic_matrix["ycb_K1"]
            
            #Prints the Meshes of the objects
            all_mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            #Only prints the Keypoints onto the image
            all_kps_p2ds = bs_utils.project_p3d(pred_kps_lst[index], 1.0, K)
            
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
            all_predictions_rgb = bs_utils.draw_p2ds(all_predictions_rgb, all_kps_p2ds, r=3, color=color)
            all_meshes_rgb = bs_utils.draw_p2ds(all_meshes_rgb, all_mesh_p2ds, r=3, color=color)

        for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
            # Index in the array, used to access the right Pose in the List
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            # Actual ID of the object in the dataset list
            obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type="ycb").copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            K = config.intrinsic_matrix["ycb_K1"]
            
            #Prints the Meshes of the objects
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            #Only prints the Keypoints onto the image
            kps_p2ds = bs_utils.project_p3d(pred_kps_lst[idx[0]], 1.0, K)
            
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
            predictions_rgb = bs_utils.draw_p2ds(predictions_rgb, kps_p2ds, r=3, color=color)
            meshes_rgb = bs_utils.draw_p2ds(meshes_rgb, mesh_p2ds, r=3, color=color)
        vis_dir = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_visuals"
        f_pth_mesh = os.path.join(vis_dir, "meshes_clean_{}.jpg".format(epoch))
        f_pth_pred = os.path.join(vis_dir, "predictions_clean_{}.jpg".format(epoch))
        f_all_pth_mesh = os.path.join(vis_dir, "meshes_all_{}.jpg".format(epoch))
        f_all_pth_pred = os.path.join(vis_dir, "predictions_all_{}.jpg".format(epoch))
        #Writes 2 images clean and full, two for the meshes and two for the predicted poses
        cv2.imwrite(f_pth_mesh, meshes_rgb)
        cv2.imwrite(f_pth_pred, predictions_rgb)
        cv2.imwrite(f_all_pth_mesh, all_meshes_rgb)
        cv2.imwrite(f_all_pth_pred, all_predictions_rgb)

        print("\n\nResults saved in {}".format(vis_dir))

# Prepare model
rndla_cfg = ConfigRandLA
config = Config()
model = FFB6D(
    n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
    n_kps=config.n_keypoints
)

PATH = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/parameters_100kps_12_9300"

if True:
    model.load_state_dict(torch.load(PATH))
else:
    ck_st = torch.load(PATH)['model_state']
    if 'module' in list(ck_st.keys())[0]:
        tmp_ck_st = {}
        for k, v in ck_st.items():
            tmp_ck_st[k.replace("module.", "")] = v
        ck_st = tmp_ck_st
    model.load_state_dict(ck_st)
model.eval()

model.to(device)

#Init Datasets 
train_data_set = train_ds = dataset_desc.Dataset('train')
test_data_set = dataset_desc.Dataset('test', DEBUG=False)

if False:
    idx = dict(
            train=0,
            val=0,
            test=0
        )

    for cat in ['test']:
        datum = test_data_set.__getitem__(idx[cat])
        idx[cat] += 1
        K = datum['K']
        cam_scale = datum['cam_scale']
        rgb = datum['rgb'].transpose(1, 2, 0)[...,::-1].copy()# [...,::-1].copy()
        for i in range(22):
            pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
            p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
            # rgb = bs_utils.draw_p2ds(rgb, p2ds)
            kp3d = datum['kp_3ds'][i]
            if kp3d.sum() < 1e-6:
                break
            kp_2ds = bs_utils.project_p3d(kp3d, cam_scale, K)
            rgb = bs_utils.draw_p2ds(
                rgb, kp_2ds, 3, bs_utils.get_label_color(datum['cls_ids'][i][0], mode=1)
            )
            ctr3d = datum['ctr_3ds'][i]
            ctr_2ds = bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
            rgb = bs_utils.draw_p2ds(
                rgb, ctr_2ds, 4, (0, 0, 255)
            )
        vis_dir = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_visuals"
        f_pth_mesh = os.path.join(vis_dir, "groundTruth_meshes.jpg")
        cv2.imwrite(f_pth_mesh, rgb)
        exit()    


#Init dataloardes https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#bacht_size = 4 and num_workes=4 might be to much (Killed)
test_dataloader = DataLoader(test_data_set, batch_size=1, shuffle=False, drop_last=True, num_workers=1)

i = 0
for batch in test_dataloader:
    i = i + 1
    #print(i)
    if i == 1:
        #change to 1 for batch to differentiate
        cal_view_pred_pose(model, batch, 0)
        break
    if i == 2:
        cal_view_pred_pose(model, batch, i)
    if i == 3:
        cal_view_pred_pose(model, batch, i)
    if i == 4:
        cal_view_pred_pose(model, batch, i)
