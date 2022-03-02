import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 1#10
lr_basemodel = 0.0001 

data_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/RSNA/checkpoints/rsna-pneumonia-detection-challenge/stage_2_train_images/'
csv_path =  '/home/skh259/LinLab/LinLab/CAT-XPLAIN/RSNA/checkpoints/rsna-pneumonia-detection-challenge/'
checkpoint_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/RSNA/checkpoints/'


# global variables
batch_size = 16
kwargs = {'batch_size':batch_size, 'num_workers':2, 'pin_memory':True}
N = 64# patch size(square patch)
M = 16# selection map size(assuming a square shaped selection map)
cls = [0,1]
eps=1e-10
loss_weight = 0.9
gpus =0
device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')

# 0: define global vars(tau, k, etc)
tau = 0.1
num_epochs = 1#10
lr = 0.0001
best_val_acc = 0
num_init = 1#3 #number of initializations of the explainer

tuning = False # Default: False. Wether to stop once the black box model is trained or not