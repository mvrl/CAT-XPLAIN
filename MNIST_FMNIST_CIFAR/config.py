import torch
from easydict import EasyDict as edict
import os
cfg = edict()

num_epochs_basemodel = 10
lr_basemodel = 0.0001

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_BASE_DIR, 'checkpoints')
checkpoint_path = data_path
plots_path = os.path.join(_BASE_DIR, 'csv_results')
log_path = os.path.join(os.path.dirname(_BASE_DIR), 'logs')

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
num_epochs = 10

lr = 0.0001
best_val_acc = 0
num_init = 3 #number of initializations of the explainer

tuning = False # Default: False. If TRUE for post-hoc experiment, the code will stop after basemodel is trained. 
                # Used to tune hyperparameters

all_metrics = True #If checkpoint selection should be done based on ace and ph_acc or all (ace,ph_acc,acc)