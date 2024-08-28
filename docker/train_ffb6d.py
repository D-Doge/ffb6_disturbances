from ffb6d import FFB6D
from common import Config, ConfigRandLA
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets.ycb.ycb_dataset as dataset_desc
from datasets.ycb.train_ycb_dataset_rgb_salt_and_pepper import Dataset_train_rgb_salt_and_pepper
from datasets.ycb.train_ycb_dataset_depth_missing_values import Dataset_depth_missing_values
from torch.utils.data import DataLoader
import numpy as np
from models.loss import OFLoss, FocalLoss
import time
import sys
import tty
import termios
import select


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_fn(data, pred):

    end_points = pred

    criterion = FocalLoss(gamma=2).to(device)
    criterion_of = OFLoss().to(device)

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

                return model(cu_dt)
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

                return model(cu_dt)

# Prepare model
rndla_cfg = ConfigRandLA
config = Config()
model = FFB6D(
    n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
    n_kps=config.n_keypoints
)

epoche = 0
iteration = 0
model_name = "depth_missing_value"
modelLoadPath = ""
#modelLoadPath = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/parameters_" + model_name + "_" + str(epoche)+ "_" + str(iteration)
if modelLoadPath != "":
    model.load_state_dict(torch.load(modelLoadPath))


model.to(device)

#Init Datasets 
train_data_set = train_ds = Dataset_depth_missing_values('train', 0)
test_data_set = dataset_desc.Dataset('test')


#Init dataloardes https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#bacht_size = 4 and num_workes=4 might be to much (Killed) -- Works on good gpu
train_dataloader = DataLoader(train_data_set, batch_size=16, shuffle=False, drop_last=False, num_workers=4)
#test_dataloader = DataLoader(test_data_set, batch_size=config.mini_batch_size, shuffle=False, drop_last=True, num_workers=1)


#Init optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)

start_time = current_time = time.time()

best_loss = 9999999
lossList = []
losstmp = []
lossAVGCount = 0
while epoche < 25:
    for batch in train_dataloader:
        model.train()
        
        # _, loss, res = self.model_fn(self.model, batch, it=it)
         
        #pred = model(data)
        pred = get_preds(model, batch)
        
        loss = loss_fn(batch, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if  loss.item() < best_loss:
            best_loss = loss.item()
            print("NEW PR!!")

    #Save every epoch
    print("Epoche: {ep}, Loss: {lo}".format(ep = epoche, lo = loss.item()))
    if((iteration % 10000 == 0) and iteration != 0):
        PATH = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/parameters_" + model_name + "_" + str(epoche)
        torch.save(model.state_dict(), PATH)
        print("model was saved")
    epoche = epoche + 1
        



PATH = "/workspace/dd6d/datasets/ycb/YCB_Video_Dataset/ffb6d_checkpoints/parameters_" + model_name + "_" + str(epoche)
torch.save(model.state_dict(), PATH)
print("model was saved")
print("FINISHED TRAINING!!")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")