import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 10
lr_basemodel = 0.0001 

data_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/MNIST_FMNIST_CIFAR/checkpoints/'
checkpoint_path = data_path



# global variables
batch_size = 64
kwargs = {'batch_size':batch_size, 'num_workers':2, 'pin_memory':True}
eps=1e-10
loss_weight = 0.9
gpus =0
device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')
dataset_class = 'full'
if dataset_class == 'full':
    num_classes = 10
else:
    num_classes = 2

N = 4
# 0: define global vars(tau, k, etc)
tau = 0.1
num_epochs = 10
lr = 0.0001
best_val_acc = 0
num_init = 3 #number of initializations of the explainer