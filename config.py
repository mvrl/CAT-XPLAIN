import torch


num_epochs_basemodel = 2
lr_basemodel = 0.0001 

data_path = './checkpoints/'
checkpoint_path = data_path
# global variables
dataset_name = 'mnist'
batch_size = 64
kwargs = {'batch_size':batch_size, 'num_workers':2, 'pin_memory':True}
eps=1e-10
loss_weight = 0.9
gpus =0
device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')
input_shape = (1,28,28)
num_classes = 2
num_patches = 10
total_num_patches = 49

# 0: define global vars(tau, k, etc)
# M*N x M*N is the size of the image
M = 7 # selection map size(assuming a square shaped selection map) 
N = 4 # patch size(square patch)
k = M*M-num_patches# number of patches for S_bar
tau = 0.1
num_epochs = 2
lr = 0.0001
best_val_acc = 0
num_init = 2 # number of initializations of the explainer
