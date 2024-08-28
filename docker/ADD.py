#!/usr/bin/env python3
import os
import cv2
import random
import torch
import numpy as np
from common import Config
from utils.basic_utils import Basic_Utils
import math

import json

#taken from basic_utlis.py
def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def cal_auc(add_dis, max_dis=0.1, name=None):
    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf;
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)


    if name is not None:
        # Save the percentage of correct poses for every 0.01 threshold increase
        threshold_increments = np.arange(0, max_dis + 0.01, 0.01)
        correct_poses_percentages = {}
        for threshold in threshold_increments:
            correct_poses_count = np.sum(D <= threshold)
            correct_poses_percentage = (correct_poses_count / n) * 100
            correct_poses_percentages[threshold] = correct_poses_percentage

        with open('/data_private/outputs/AUC_'+name+'.json', 'w') as json_file:
            json.dump(correct_poses_percentages, json_file, indent=4)
    
    return aps * 100

def cal_add_cuda(
    pred_RT, gt_RT, p3ds
):
    pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
    return torch.mean(dis)

def cal_adds_cuda(
    pred_RT, gt_RT, p3ds
):
    N, _ = p3ds.size()
    pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    pd = pd.view(1, N, 3).repeat(N, 1, 1)
    gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
    gt = gt.view(N, 1, 3).repeat(1, N, 1)
    dis = torch.norm(pd - gt, dim=2)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis)


#taken from pvn3d_eval_utlis_kpls.py
def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, 
    gt_kps, gt_ctrs, pred_kpc_lst
):
    config = Config(ds_name='ycb')
    bs_utils = Basic_Utils(config)
    cls_lst = config.ycb_cls_lst

    
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis, cls_kp_err)

def tensor_to_numpy(tensor_or_array):
    """Converts a PyTorch tensor or a NumPy array to a NumPy array.

    Args:
        tensor_or_array (torch.Tensor or np.ndarray): Input tensor or array.

    Returns:
        np.ndarray: NumPy array representing the input data.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        return tensor_or_array.cpu().numpy() if tensor_or_array.is_cuda else tensor_or_array.numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        # Input is already a NumPy array, so return as is
        return tensor_or_array
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array")

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose given by a 3x1 vector.
    :param t_gt: Translation element of the ground truth pose given by a 3x1 vector.
    :return: Error of t_est w.r.t. t_gt.
    """

    t_est = tensor_to_numpy(t_est)
    t_gt = tensor_to_numpy(t_gt)

    if not(t_est.size == t_gt.size == 3):
        return 3.5 # max value
    error = np.linalg.norm(t_gt - t_est)
    return error

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose given by a 3x3 matrix.
    :param R_gt: Rotational element of the ground truth pose given by a 3x3 matrix.
    :return: Error of t_est w.r.t. t_gt.
    """
    R_gt = tensor_to_numpy(R_gt)
    R_est = tensor_to_numpy(R_est)
    assert(R_est.shape == R_gt.shape == (3, 3))
    try:
        error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    except np.linalg.LinAlgError as e:
        if np.all(R_gt == 0):
            return math.pi
        print(R_gt)
        exit()
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    return error

def get_RE_TE(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, 
    gt_kps, gt_ctrs, pred_kpc_lst
):
    config = Config(ds_name='ycb')

    
    n_cls = config.n_classes
    te_list = [list() for i in range(n_cls)]
    re_list = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
        gt_RT = RTs[icls].cpu().detach().numpy()

        #print(gt_RT[:, 3])
        #print(pred_RT[:, 3])



        te_list.append(te(gt_RT[:, 3], pred_RT[:, 3]))
        re_list.append(re(gt_RT[:, :3], pred_RT[:, :3]))

    return (re_list, te_list)


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
        if (len(rotation_invarant_list) >= 2):
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
    

def get_rotations_for_cls(cls_id):
    if cls_id == 13: #case 24_bowl
        return (None, [(0,0,1)]) # rotation invariance around z axis
    
    if cls_id == 16: #case 36_wood_block
        symmertic_dict = {} # is treated like a quader
        symmertic_dict[(0.02982577, -0.01951606, -0.99936457)] = [np.radians(0), np.radians(90), np.radians(180), np.radians(270)]
        symmertic_dict[(-0.72384695,  0.68906924, -0.03505948)] = [np.radians(0), np.radians(180)]
        return(symmertic_dict, None)
    
    if cls_id == 19: #case 051_large_clamp
        symmertic_dict = {}
        symmertic_dict[(0.11252108, -0.99226705, -0.05239371)] = [np.radians(0), np.radians(180)]
        return(symmertic_dict, None)
    
    if cls_id == 20: #case 052extra__large_clamp
        symmertic_dict = {}
        symmertic_dict[(-0.99088365, -0.13469705, -0.00251033)] = [np.radians(0), np.radians(180)]
        return(symmertic_dict, None)
    
    if cls_id == 21: #case 061_foam_brick
        symmertic_dict = {}
        symmertic_dict[(0,0,1)] = [np.radians(0), np.radians(180)] #is not treated as a quader, because of the 3 holes
        return(symmertic_dict, None)
    
    return (None, None)

def error_metric(R_gt, t_gt, R_est, t_est, cls_id):
    if R_gt is None: #Case False dection
        return math.inf
    if R_est is None: #Case Object was not dected
        return math.inf

    symmertic_dict, rotation_invarant_list = get_rotations_for_cls(cls_id)
    rotation_error = rotation_error_metric(R_gt, R_est, symmertic_dict, rotation_invarant_list) / math.pi # Scale rotation error to be 0 <= rotation_error <= 1, so it has the same weight as the translation error
    translation_error = te(t_est, t_gt)
    translation_error = min(translation_error/0.1, 1) #normalize the translation error it should be 1 at 3.5m, since it is the max range of asus xtion depth sensor, larger error do not make much sence

    return rotation_error + translation_error

def dd_score(erro_list):
    sum = 0
    for error in erro_list:
        if error == math.inf: #Case wrong dection add nothing to the sum
            continue

        sum = sum + (1/(1+error)) #add 1 if pereft pose, add 0.33 in worst case

    return sum / len(erro_list) #Normilze between 0 and 1

def get_dd_error_over_real_objects(cls_ids, pred_pose_lst, pred_cls_ids, RTs, 
    gt_kps, gt_ctrs, pred_kpc_lst
):
    config = Config(ds_name='ycb')
    n_cls = config.n_classes
    error_list = []
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break


        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0: #case class was not predicted
            pred_RT = None
            error_list.append(math.inf)
            continue
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
        gt_RT = RTs[icls].cpu().detach().numpy()

        R_gt = gt_RT[:, :3]
        t_gt = gt_RT[:, 3]
        R_est = pred_RT[:, :3]
        t_est = pred_RT[:, 3]
        error_list.append(error_metric(R_gt, t_gt, R_est, t_est, cls_id))

    return error_list


############### Unfinished
def get_dd_error_over_detected(cls_ids, pred_pose_lst, pred_cls_ids, RTs, 
    gt_kps, gt_ctrs, pred_kpc_lst
):
    config = Config(ds_name='ycb')
    n_cls = config.n_classes
    error_list = []
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break


        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0: #case class was not predicted
            pred_RT = None
            error_list.append(math.inf)
            continue
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
        gt_RT = RTs[icls].cpu().detach().numpy()

        R_gt = gt_RT[:, :3]
        t_gt = gt_RT[:, 3]
        R_est = pred_RT[:, :3]
        t_est = pred_RT[:, 3]
        error_list.append(error_metric(R_gt, t_gt, R_est, t_est, cls_id))

    return error_list


def get_dd_error_over_real_and_predicted(cls_ids, pred_pose_lst, pred_cls_ids, RTs, 
    gt_kps, gt_ctrs, pred_kpc_lst
):
    # print("--------------------------dd-------------------")
    # print("cls_ids")
    # print(cls_ids.shape)
    # print("----------------------------------")
    # print("pred_pose_lst")
    # print(pred_pose_lst)
    # print("----------------------------------")
    # print("pred_cls_ids")
    # print(pred_cls_ids)
    # print("----------------------------------")
    # print("RTs")
    # print(RTs)
    # print("------------End----------dd")

    pred_cls_ids_list = pred_cls_ids.tolist()

    config = Config(ds_name='ycb')
    n_cls = config.n_classes
    error_list = []
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break


        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0: #case class was not predicted
            pred_RT = None
            error_list.append(math.inf)
            continue
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
        gt_RT = RTs[icls].cpu().detach().numpy()

        R_gt = gt_RT[:, :3]
        t_gt = gt_RT[:, 3]
        R_est = pred_RT[:, :3]
        t_est = pred_RT[:, 3]
        error_list.append(error_metric(R_gt, t_gt, R_est, t_est, cls_id))
        pred_cls_ids_list.remove(cls_id)

    for item in pred_cls_ids_list: # Every detected object, that is not realy here has and infinte error
        error_list.append(math.inf)
    return error_list