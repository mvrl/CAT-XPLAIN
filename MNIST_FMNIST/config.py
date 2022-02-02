import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 2
lr_basemodel = 0.0001 

data_path = '/u/amo-d0/grad/skh259/projects/CAT-XPLAIN/MNIST_FMNIST/checkpoints/'
checkpoint_path = data_path



# global variables
batch_size = 64
kwargs = {'batch_size':batch_size, 'num_workers':2, 'pin_memory':True}
eps=1e-10
loss_weight = 0.9
gpus =0
device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')
input_shape = (1,28,28)
input_dim = 28
num_classes = 2
total_num_patches = 7*7

# 0: define global vars(tau, k, etc)
# M*N x M*N is the size of the image
M = 7 # selection map size(assuming a square shaped selection map) 
N = 4 # patch size(square patch)
tau = 0.1
num_epochs = 2
lr = 0.0001
best_val_acc = 0
num_init = 2 # number of initializations of the explainer


## For MRI data
#cfg.mri_data = '/u/vul-d1/scratch/subash/ADNI/Preprocessed'
cfg.mri_data = '/mnt/gpfs2_16m/pscratch/nja224_uksr/xin_data/Preprocessed/'
cfg.checkpoint = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/CAT-XPLAIN/MRI/checkpoints/'
cfg.test_csv = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/CAT-XPLAIN/MRI/cv_paths/CN_AD/CN_ADcombined_test_list.csv'
cfg.folds = '/mnt/gpfs2_16m/pscratch/nja224_uksr/SKH259/LinLab/CAT-XPLAIN/MRI/cv_paths/'
