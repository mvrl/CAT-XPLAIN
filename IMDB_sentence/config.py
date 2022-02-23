import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 20
lr_basemodel = 0.0001 

data_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/IMDB_sentence/checkpoints/'
checkpoint_path = data_path
imdb_data_path = '/home/skh259/LinLab/LinLab/Datasets/'



# global variables
batch_size = 64
kwargs = {'batch_size':batch_size, 'num_workers':2, 'pin_memory':True}
eps=1e-10
loss_weight = 0.9
gpus =0
device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')
max_length = 50
num_classes = 2
emb_dim = 384

tau = 0.1
num_epochs = 20
lr = 0.0001
best_val_acc = 0
num_init = 3 # number of initializations of the explainer
