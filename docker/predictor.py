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
import datasets.ycb.ycb_dataset as dataset_desc
from torch.utils.data import DataLoader
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config(ds_name="ycb")

bs_utils = Basic_Utils(config)

class Predictor:
    def __init__(self):
        # Prepare model
        rndla_cfg = ConfigRandLA
        config = Config()
        self.model = FFB6D(
            n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
            n_kps=config.n_keypoints
        )

        PATH = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/FFB6D_best.pth.tar"

        if False:
            self.model.load_state_dict(torch.load(PATH))
        else:
            ck_st = torch.load(PATH, map_location=device)['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            self.model.load_state_dict(ck_st)
        self.model.eval()

        self.model.to(device)

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_data(self, rgb, depth, K, cam_scale=10000):
        dpt_um = np.array(depth)
        if len(dpt_um.shape) == 3:
            dpt_um = dpt_um[:, :, 0]
        print(dpt_um.shape)

        rgb = np.array(rgb)[:, :, :3]
        msk_dp = dpt_um > 1e-6

        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )

        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)


        h = rgb.shape[0]
        w = rgb.shape[1]
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }
        sr2msk = {
            pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
        )
        item_dict.update(inputs)

        for key in item_dict:
            item_dict[key] = np.expand_dims(item_dict[key], axis=0)

        return item_dict
    
    def make_prediction(self, rgb_image, depth_image, K, cam_scale=10000):
        data = self.get_data(rgb_image, depth_image, K, cam_scale)

        if device != 'cpu':
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
                end_points = self.model(cu_dt)
                _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        else:
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

                
                end_points = self.model(cu_dt)
                _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)


        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        pred_cls_ids, pred_pose_lst, pred_kps_lst = cal_frame_poses(
            pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
            end_points['pred_kp_ofs'][0], True, config.n_objects, True,
            None, None
        )

        return pred_cls_ids, pred_pose_lst, pred_kps_lst

