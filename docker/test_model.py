from ffb6d import FFB6D
from common import Config, ConfigRandLA
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets.ycb.ycb_dataset as dataset_desc
from torch.utils.data import DataLoader
import numpy as np
from models.loss import OFLoss, FocalLoss
import time
import sys
import tty
import termios
import select
from utils.pvn3d_eval_utils_kpls import cal_frame_poses
from ADD import eval_metric, cal_auc
import json
import math

from datasets.ycb.ycb_dataset_brightness import Dataset_Brightness
from datasets.ycb.ycb_dataset_depth_missing_values import Dataset_depth_missing_values
from datasets.ycb.ycb_dataset_rgb_salt_and_pepper import Dataset_rgb_salt_and_pepper
from datasets.ycb.ycb_dataset_rgb_missing_values import Dataset_rgb_missing_values
from datasets.ycb.ycb_dataset_rgb_cicle_spot_blur import Dataset_rgb_circle_spot_blur
from datasets.ycb.ycb_dataset_depth_cirlce_spot_missing import Dataset_depth_missing_cricle
from datasets.ycb.ycb_dataset_rgb_cicle_spot_missing import Dataset_rgb_circle_spot_missing
from datasets.ycb.ycb_dataset_depth_noise import Dataset_depth_noise
from datasets.ycb.ycb_dataset_rgb_noise import Dataset_rgb_noise
from datasets.ycb.ycb_dataset_rgb_motion_blur import Dataset_rgb_motion_blur
from datasets.ycb.ycb_dataset_depth_motion_blur import Dataset_depth_motion_blur

from ADD import get_RE_TE, get_dd_error_over_real_and_predicted, get_dd_error_over_real_objects, dd_score

import itertools

start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_fn(data, pred):

    end_points = pred

    criterion = FocalLoss(gamma=2).to(device)
    criterion_of = OFLoss().to(device)

    if torch.cuda.is_available():
        #Send tensors to GPU
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
    else:
        #Dont send tensors to GPU
        with torch.set_grad_enabled(False):
                cu_dt = {}
                # device = torch.device('cuda:{}'.format(args.local_rank))
                for key in data.keys():
                    if data[key].dtype in [np.float32, np.uint8]:
                        cu_dt[key] = torch.from_numpy(data[key].astype(np.float32))
                    elif data[key].dtype in [np.int32, np.uint32]:
                        cu_dt[key] = torch.LongTensor(data[key].astype(np.int32))
                    elif data[key].dtype in [torch.uint8, torch.float32]:
                        cu_dt[key] = data[key].float()
                    elif data[key].dtype in [torch.int32, torch.int16]:
                        cu_dt[key] = data[key].long()


    labels = cu_dt['labels']
    loss_rgbd_seg = criterion(
        end_points['pred_rgbd_segs'], labels.view(-1)
    ).sum()
    loss_kp_of = criterion_of(
        end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
    ).sum()
    loss_ctr_of = criterion_of(
        end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
    ).sum()

    loss_lst = [
        (loss_rgbd_seg, 2.0), (loss_kp_of, 1.0), (loss_ctr_of, 1.0),
    ]
    loss = sum([ls * w for ls, w in loss_lst])
    return loss

def get_preds(model, data):
    
    if torch.cuda.is_available():
        #Send tensors to GPU
        with torch.set_grad_enabled(True):
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

                return (model(cu_dt), cu_dt)
    else:
        #Dont send tensors to GPU
        with torch.set_grad_enabled(True):
                cu_dt = {}
                # device = torch.device('cuda:{}'.format(args.local_rank))
                for key in data.keys():
                    if data[key].dtype in [np.float32, np.uint8]:
                        cu_dt[key] = torch.from_numpy(data[key].astype(np.float32))
                    elif data[key].dtype in [np.int32, np.uint32]:
                        cu_dt[key] = torch.LongTensor(data[key].astype(np.int32))
                    elif data[key].dtype in [torch.uint8, torch.float32]:
                        cu_dt[key] = data[key].float()
                    elif data[key].dtype in [torch.int32, torch.int16]:
                        cu_dt[key] = data[key].long()

                return (model(cu_dt), cu_dt)

def get_poses_from_predictions(predictons, cu_dt):
    end_points = predictons
    _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)


    pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
    pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
        pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0].detach(),
        end_points['pred_kp_ofs'][0].detach(), True, config.n_objects, True,
        None, None
    )

    return pred_cls_ids, pred_pose_lst, pred_kps_lst

# Prepare model
rndla_cfg = ConfigRandLA
config = Config()
model = FFB6D(
    n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
    n_kps=config.n_keypoints
)

epoche = 0
iteration = 0
model_name = "100kps"
modelLoadPath = ""
#modelLoadPath = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/parameters_" + model_name + "_" + str(epoche)+ "_" + str(iteration)
PATH = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/FFB6D_best.pth.tar"
ck_st = torch.load(PATH)['model_state']
if 'module' in list(ck_st.keys())[0]:
    tmp_ck_st = {}
    for k, v in ck_st.items():
        tmp_ck_st[k.replace("module.", "")] = v
    ck_st = tmp_ck_st
model.load_state_dict(ck_st)
model.eval()

model.to(device)


model.to(device)

metrics = {}

values = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for value in  values:

    #Init Datasets 
    #test_data_set = dataset_desc.Dataset('test')
    #test_data_set = Dataset_Brightness("test", 0)
    #test_data_set = Dataset_depth_missing_values("test", value)
    #test_data_set = Dataset_rgb_salt_and_pepper("test", 20)
    #test_data_set = Dataset_rgb_missing_values("test", value)
    #test_data_set = Dataset_rgb_circle_spot_blur("test", value, 50, 100, 10, 30)
    #test_data_set = Dataset_depth_missing_cricle("test", value, 50, 100)
    #test_data_set = Dataset_rgb_circle_spot_missing("test", value, 50, 100)
    #test_data_set = Dataset_depth_noise("test", value)
    #test_data_set = Dataset_rgb_noise("test", value)
    #test_data_set = Dataset_rgb_motion_blur("test", value)
    test_data_set = Dataset_depth_motion_blur("test", value)


    #Init dataloardes https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    #bacht_size = 4 and num_workes=4 might be to much (Killed) -- Works on good gpu

    #TODO: Change to new dataloader that can add other disturbances
    batch_size = 1
    test_dataloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)




    count = 0
    loss_avg = 0

    all_cls_ids = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21) #TODO innit with the clas ids
    metric_dict = {} #TODO innit with the lists
    for cls_id in all_cls_ids:
        #                       add     adds    kp_error
        metric_dict[cls_id] = (list(), list(), list())

    te_list = []
    re_list = []
    dd_error_list_over_real_and_detected = []
    dd_error_list_over_real = []

    for batch in test_dataloader:  
        #pred = model(data)
        pred, cu_dt = get_preds(model, batch)

        loss = loss_fn(batch, pred)
        loss_avg += loss.item() / batch_size

        pred_cls_ids, pred_pose_lst, pred_kps_lst = get_poses_from_predictions(pred, cu_dt)

        #TODO: give back the avarge ADD/S and the AOC. 
        cls_add_dis, cls_adds_dis, cls_kp_err = eval_metric(cu_dt["cls_ids"][0], pred_pose_lst, pred_cls_ids, cu_dt["RTs"][0], cu_dt['kp_targ_ofst'][0], cu_dt['ctr_targ_ofst'][0], pred_kps_lst)
        rotation_error_list, translation_error_list = get_RE_TE(cu_dt["cls_ids"][0], pred_pose_lst, pred_cls_ids, cu_dt["RTs"][0], cu_dt['kp_targ_ofst'][0], cu_dt['ctr_targ_ofst'][0], pred_kps_lst)
        te_list = te_list + translation_error_list
        re_list = re_list + rotation_error_list

        dd_error_list_over_real_and_detected = dd_error_list_over_real_and_detected + get_dd_error_over_real_and_predicted(cu_dt["cls_ids"][0], pred_pose_lst, pred_cls_ids, cu_dt["RTs"][0], cu_dt['kp_targ_ofst'][0], cu_dt['ctr_targ_ofst'][0], pred_kps_lst)
        dd_error_list_over_real = dd_error_list_over_real + get_dd_error_over_real_objects(cu_dt["cls_ids"][0], pred_pose_lst, pred_cls_ids, cu_dt["RTs"][0], cu_dt['kp_targ_ofst'][0], cu_dt['ctr_targ_ofst'][0], pred_kps_lst)

        #We propaply need a dict that has an entry for every class, that contains a 3 lists, one for add, adds and kp_error
        for cls_id in cu_dt["cls_ids"][0]:
            #0 might be the background?
            if cls_id == 0:
                continue
            add_list, adds_list, kp_error_list = metric_dict[cls_id.item()]

            add_list.append(cls_add_dis[cls_id.item()])
            adds_list.append(cls_adds_dis[cls_id.item()])
            kp_error_list.append(cls_kp_err[cls_id.item()])
            

        count += 1
        #print(count)
        #if count > 10:
        #    break

    loss_avg = loss_avg / count

    add_list = list()
    adds_list = list()
    # Keep in mind, that here the classes are dived into ADD and ADDs
    for cls_id in all_cls_ids:
        if cls_id in config.ycb_sym_cls_ids:
            adds_list = adds_list + metric_dict[cls_id][1]
        else:
            add_list = add_list + metric_dict[cls_id][0]

    add_list = np.array([x[0] for x in add_list])
    adds_list = np.array([x[0] for x in adds_list])

    add_auc = cal_auc(add_list, name="ADD")
    adds_auc = cal_auc(adds_list, name="ADDS")

    add_avg = sum(add_list) / len(add_list)
    adds_avg = sum(adds_list) / len(adds_list)

    dd_error_score_over_real = dd_score(dd_error_list_over_real)
    dd_error_score_over_real_and_detected = dd_score(dd_error_list_over_real_and_detected)

    re_list = [item for item in re_list if isinstance(item, (int, float))]
    re_avg = sum(re_list) / len(re_list)
    re_variance = sum((re - re_avg) ** 2 for re in re_list) / len(re_list)
    # Step 3: Calculate the standard deviation
    re_std_dev = math.sqrt(re_variance)


    te_list = [item for item in te_list if isinstance(item, (int, float))]
    te_avg = sum(te_list) / len(te_list)
    te_variance = sum((te - te_avg) ** 2 for te in te_list) / len(te_list)
    # Step 3: Calculate the standard deviation
    te_std_dev = math.sqrt(te_variance)

    metrics[value] = (add_auc, adds_auc, loss_avg, add_avg, adds_avg, dd_error_score_over_real, dd_error_score_over_real_and_detected, re_avg, re_std_dev, te_avg, te_std_dev, len(dd_error_list_over_real_and_detected))
    print("------------------------------------------------------------")
    print(value)
    print("ADD " + str(add_auc))
    print("ADDS " + str(adds_auc))
    print("Loss_avg " + str(loss_avg))
    print("ADD_avg " + str(add_avg))
    print("ADDS_avg " + str(adds_avg))
    print("dd_error_score_over_real " + str(dd_error_score_over_real))
    print("dd_error_score_over_real_and_detected " + str(dd_error_score_over_real_and_detected))
    print("re_avg " + str(re_avg))
    print("te_avg " + str(te_avg))
    print("------------------------------------------------------------")

print("ADD_AUC, ADDS_ AUC, AVG_LOSS, AVG_ADD, AVG_ADDS, dd_error_score_over_real, dd_error_score_over_real_and_detected, re_avg, te_avg")
print(metrics)
# Save dictionary to a JSON file
with open('depth_motion_blur.json', 'w') as json_file:
    json.dump(metrics, json_file, indent=4)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")