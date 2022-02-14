# ## Most of the code below is taken from: https://github.com/pranoy-panda/Causal-Feature-Subset-Selection

from models import ViT, initialize_model
import itertools
from easydict import EasyDict as edict

# relevant libraries
import numpy as np
import torch
import torchvision
import torch.utils.data as data
from mri_dataloader import prep_data, Dataset_MRI
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import random
from joblib import dump, load
from tqdm import tqdm
from utils import sample_concrete, custom_loss, generate_xs, metrics, imgs_with_random_patch_generator_mri
from config import *
import os
from utils import train_basemodel, test_basemodel

# for supressing warnings
import warnings
warnings.filterwarnings('ignore')


def seed_initialize(seed = 12345):
  ######
  # set the seeds for reproducibility
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  SEED = seed
  random.seed(SEED) 
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)


def train_eval(dataset_name,view_type,bb_model_type,sel_model_type,depth,dim,num_patches,validation='without_test'):
  seed_initialize(seed = 12345)
  groups = 'CN_AD'
  cls = [0,1]
  M = 19
  N = 10
  input_shape = (1,190,190)
  input_dim = 190
  LABEL_PATH = cfg.folds
  TEST_LABEL = cfg.test_csv
  checkpoint_path = cfg.checkpoint
  
  
  print("For dataset:",dataset_name)
  print("For viewtype:",view_type)
  print("For Experiment with bb_model:",bb_model_type)
  print("For Experiment with sel_model:",sel_model_type)
  print('1. Training the Basemodel...... \n')

  num_patches = int(num_patches*M*M)
  k = M*M-int(num_patches)# number of patches for S_bar
  ## Initialize Base model
  test_acc_list = []
  test_ice_list = []

  val_acc_list = []
  val_ice_list = []

  for iter_num in range(num_init):
    batch_size = 16 #Not 64 like for MNIST of FMNIST
    TEST_NUM = iter_num
    exper_path = os.path.join(checkpoint_path,'iter'+str(iter_num))
    if not os.path.exists(exper_path):
        os.makedirs(exper_path)
    TRAIN_LABEL, VAL_LABEL = prep_data(LABEL_PATH ,exper_path,TEST_NUM, groups)
    train_dataset = Dataset_MRI(label_file=TRAIN_LABEL,groups='CN_AD',random_patch=False,M=M,N=N,num_patches=num_patches)
    trainloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = Dataset_MRI(label_file=VAL_LABEL,groups='CN_AD',random_patch=False,M=M,N=N,num_patches=num_patches)
    valloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, drop_last=True)

    test_dataset = Dataset_MRI(label_file=TEST_LABEL,groups='CN_AD',random_patch=False,M=M,N=N,num_patches=num_patches)
    testloader = torch.utils.data.DataLoader(test_dataset, num_workers=8, batch_size=batch_size, shuffle=False, drop_last=True)

    bb_model = initialize_model(bb_model_type,num_classes=2,input_dim=input_dim,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device).float()  
    
    LossFunc_basemodel = torch.nn.CrossEntropyLoss(size_average = True)
    optimizer_basemodel = torch.optim.Adam(bb_model.parameters(),lr = lr_basemodel) 

    # train the basemodel 
    bb_model = train_basemodel(dataset_name,cls,trainloader,
                  valloader,
                  bb_model,
                  LossFunc_basemodel,
                  optimizer_basemodel,
                  num_epochs_basemodel,
                  batch_size,
                  checkpoint_path)

    # testing the model on held-out validation dataset
    if validation == 'without_test':
      test_basemodel(cls,valloader,bb_model)
    else:
      test_basemodel(cls,testloader,bb_model)
    print('Basemodel trained! \n')

    ##############################################################

    print('2. Starting main feature selection algorithm......... \n')

    '''
    generate images with random patches selected for calculating the ACE metric
    '''
    val_dataset_random = Dataset_MRI(label_file=VAL_LABEL,groups='CN_AD',view_type=view_type,random_patch=True,M=M,N=N,num_patches=num_patches)
    imgs_with_random_patch_val = torch.utils.data.DataLoader(val_dataset_random, num_workers=8, batch_size=batch_size, shuffle=False, drop_last=True)
    imgs_with_random_patch_val = imgs_with_random_patch_generator_mri(imgs_with_random_patch_val,len(val_dataset_random),num_patches)

    test_dataset_random = Dataset_MRI(label_file=TEST_LABEL,groups='CN_AD',view_type=view_type,random_patch=True,M=M,N=N,num_patches=num_patches)
    imgs_with_random_patch_test = torch.utils.data.DataLoader(test_dataset_random, num_workers=8, batch_size=batch_size, shuffle=False, drop_last=True)
    imgs_with_random_patch_test = imgs_with_random_patch_generator_mri(imgs_with_random_patch_test,len(test_dataset_random),num_patches)

    # training loop where we run the experiments for multiple times and report the 
    # mean and standard deviation of the metrics ph_acc and ICE.
    ## Initialize Selection model
    selector = initialize_model(sel_model_type,num_classes=M*M,input_dim=input_dim,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device).float()
    #optimizer
    optimizer = torch.optim.Adam(selector.parameters(),lr = lr)
    
    val_accs = []
    val_ices = []
    # training loop
    for epoch in range(num_epochs):  
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            X, Y = data
            batch_size = X.size(0)
            X = X.to(device)
            Y = (Y == cls[-1]).long().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # 1: get the logits from construct_gumbel_selector()
            logits = selector.forward(X)
            # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick      
            selected_subset = sample_concrete(tau,k,logits,train=True) # get S_bar from explainer
            # 3: reshape selected_subset to the size M x M i.e. the size of the patch or superpixel
            selected_subset = torch.reshape(selected_subset,(batch_size,M,M))
            selected_subset = torch.unsqueeze(selected_subset,dim=1)
            # 4: upsampling the selection map
            upsample_op = nn.Upsample(scale_factor=N, mode='nearest')
            v = upsample_op(selected_subset)
            # 5: X_Sbar = elementwise_multiply(X,v); compute f_{bb}(X_Sbar)
            X_Sbar = torch.mul(X,v) # output shape will be [batch_size,1,M*N,M*N]
            
            f_xsbar = F.softmax(bb_model(X_Sbar)) # f_xs stores p(y|xs)
            with torch.no_grad():
              f_x =  F.softmax(bb_model(X)) # f_x stores p(y|x)          

            # optimization function
            loss = custom_loss(f_xsbar,f_x,batch_size)

            loss.backward()
            optimizer.step()
          
            running_loss+=loss.item() # average loss per sample
  ################################################################################
        val_acc,val_ice = metrics(cls,selector,k,M,N,iter_num,valloader,imgs_with_random_patch_val,bb_model)
        val_accs.append(val_acc)
        val_ices.append(val_ice)
        if not os.path.exists(checkpoint_path):
          os.makedirs(checkpoint_path)
        model_checkpoint = os.path.join(checkpoint_path,dataset_name+str(iter_num)+'_'+str(epoch)+'_posthoc_selector.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': selector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_checkpoint)     
          # print loss and validation accuracy at the end of each epoch 
        print('\nInitialization Number %d-> epoch: %d, average loss: %.3f, val_acc: %.3f, ICE: %.3f \n' %(iter_num+1, epoch + 1,running_loss/i, val_acc, val_ice))

    best_val_performance = [(val_accs[item]+val_ices[item])/2 for item in range(len(val_ices))]
    best_epoch = np.argmax(best_val_performance)
    val_acc_list.append(val_accs[best_epoch])
    val_ice_list.append(val_ices[best_epoch])
    print("BEST EPOCH BASED ON VAL PERFORMANCE:",best_epoch)
    print("BEST (VAL_ACC,VAL_ICE)",(val_accs[best_epoch],val_ices[best_epoch]))
    
    best_model_path = os.path.join(checkpoint_path,dataset_name+str(iter_num)+'_'+str(best_epoch)+'_posthoc_selector.pt')
    ## Initialize Selection model
    best_model = initialize_model(sel_model_type,num_classes=M*M,input_dim=input_dim,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device).float()
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    ## Initialize base blackbox model
    bb_checkpoint = torch.load(checkpoint_path+'/'+dataset_name+'_model.pt')
    bb_model = initialize_model(bb_model_type,num_classes=2,input_dim=input_dim,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device).float()  
    bb_model.load_state_dict(bb_checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_acc,test_ice = metrics(cls,best_model,k,M,N,iter_num,testloader,imgs_with_random_patch_test,bb_model,intrinsic=False)
    
    if validation == 'with_test':
      print('test (ph acc, ICE):',(test_acc,test_ice))

    test_acc_list.append(test_acc)
    test_ice_list.append(test_ice)
   
  print('mean val ph acc: %.3f'%(np.mean(val_acc_list)),', std dev: %.3f '%(np.std(val_acc_list))) 
  print('mean val ICE: %.3f'%(np.mean(val_ice_list)),', std dev: %.3f '%(np.std(val_ice_list))) 

  if validation == 'with_test':
    print('mean test ph acc: %.3f'%(np.mean(test_acc_list)),', std dev: %.3f '%(np.std(test_acc_list))) 
    print('mean test ICE: %.3f'%(np.mean(test_ice_list)),', std dev: %.3f '%(np.std(test_ice_list))) 

  print('\nDONE! \n')

if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',  type=str,help="Dataset type: Options:[mri]", default= 'mri')
    parser.add_argument('--view_type', type=str, help='View type either for Single View or MultiView Options: [0 or 1 or 2 or multi]', default='1')
    parser.add_argument('--bb_model_type', type=str,help="Base_model type: Options:[ViT]",default="ViT")
    parser.add_argument('--sel_model_type', type=str,help="select_model type: Options:[ViT]",default="ViT")
    parser.add_argument('--depth',  type=str,help="depth of the transformer block: Options[1,2,4,8,10]", default= "8")
    parser.add_argument('--dim',  type=str,help="dimension of internal state: Options[64,128,256,512]", default= "128")
    parser.add_argument('--num_patches',  type=str,help="frac for number of patches to select: Options[0.05,0.10,0.25,0.50,0.75]", default= "0.25")
    parser.add_argument('--validation', type=str,help=" Perform validation on validation or test set: Options:[without_test, with_test]",default="with_test")
    parser.add_argument('--sweep', type=str,help="select_model type: Options:[sweep,no_sweep]",default="no_sweep")
    args = parser.parse_args()
    validation = args.validation
    num_patches = float(args.num_patches)
    dataset_name = args.dataset_name
    view_type = args.view_type
    bb_model_type = args.bb_model_type
    sel_model_type = args.sel_model_type
    depth = int(args.depth)
    dim = int(args.dim)
   
    train_eval(dataset_name=dataset_name,view_type=view_type,
            bb_model_type=bb_model_type,sel_model_type=sel_model_type,
            depth=depth,dim=dim,num_patches=num_patches,validation=validation)
