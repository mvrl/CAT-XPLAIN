import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 2#10
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

N = 4
# 0: define global vars(tau, k, etc)
tau = 0.1
num_epochs = 2#10

train_further = True
post_num_epochs = 2#5

lr = 0.0001
best_val_acc = 0
num_init = 1#3 #number of initializations of the explainer

tuning = False # Default: False. Wether to stop once the black box model is trained or not