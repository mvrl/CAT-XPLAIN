### Most of the code below is taken from: https://github.com/pranoy-panda/Causal-Feature-Subset-Selection

# relevant libraries
import numpy as np
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets as vision_datasets
from torchtext import datasets as text_datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import random
from joblib import dump, load
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from models import modifiedViT
from utils import sample_concrete, custom_loss, generate_xs, metrics, imgs_with_random_patch_generator, load_dataset
from config import *
import os

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


# for supressing warnings
import warnings
warnings.filterwarnings('ignore')


print(np.version.version)
print(torch.__version__)
print(device)

val_acc_list = []
val_ice_list = []

test_acc_list = []
test_ice_list = []


def train_eval(dataset_name,loss_weight,num_patches,validation):
  seed_initialize(seed = 12345)
  k = M*M-num_patches# number of patches for S_bar
  ###################################### LOAD DATASET ######################################################
  cls, trainloader, valloader, testloader, train_datasize, valid_datasize, test_datasize = load_dataset(dataset_name=dataset_name)

  ##########################################################################################################################################################
  ################################################# RANDOM PATCH SELECTED DATASET CREATOR ##################################################################
  '''
  to generate images with random patches selected
  '''

  imgs_with_random_patch_val = imgs_with_random_patch_generator(valloader,valid_datasize,num_patches)
  imgs_with_random_patch_test = imgs_with_random_patch_generator(testloader,test_datasize,num_patches)

  ######################################################################################################################################################
  #                                             MAIN TRAINING CODE                                                                                     #
  ######################################################################################################################################################

  # training loop where we run the experiments for multiple times and report the 
  # mean and standard deviation of the metrics ph_acc and ICE.
  for iter_num in range(num_init):
      
      # intantiating the interpretable transformer
      bb_model = modifiedViT(
              image_size = 28,
              patch_size = 4,
              num_classes = num_classes,
              channels = 1,
              dim = 128,
              depth = 2,
              heads = 4,
              mlp_dim = 256,
              dropout = 0.1,
              emb_dropout = 0.1,
              explain = True).to(device)
      selector = bb_model
      LossFunc = torch.nn.CrossEntropyLoss(size_average = True)
      #optimizer
      optimizer = torch.optim.Adam(selector.parameters(),lr = lr)
      # variable for keeping track of best ph_acc across different iterations 
      val_accs = []
      val_ices = []

      # training loop
      for epoch in range(num_epochs):  
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
        # get the inputs
          X, Y = data
          X = X.to(device)
          Y = (Y == cls[-1]).long().to(device)
          batch_size = X.size(0)
        # zero the parameter gradients
          optimizer.zero_grad()
        # 1: get the logits from construct_gumbel_selector()
          class_logits, patch_logits = selector.forward(X)
        # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick      
          selected_subset = sample_concrete(tau,k,patch_logits,train=True) # get S_bar from explainer
        # 3: reshape selected_subset to the size M x M i.e. the size of the patch or superpixel
          selected_subset = torch.reshape(selected_subset,(batch_size,M,M))
          selected_subset = torch.unsqueeze(selected_subset,dim=1)
        # 4: upsampling the selection map
          upsample_op = nn.Upsample(scale_factor=N, mode='nearest')
          v = upsample_op(selected_subset)
        # 5: X_Sbar = elementwise_multiply(X,v); compute f_{bb}(X_Sbar)
          X_Sbar = torch.mul(X,v) # output shape will be [batch_size,1,M*N,M*N]

          f_xsbar = F.softmax(bb_model(X_Sbar)[0]) # f_xs stores p(y|xs)
          with torch.no_grad():
              f_x =  F.softmax(class_logits) # f_x stores p(y|x)          

        # optimization function
          loss = custom_loss(f_xsbar,f_x,batch_size)
          loss = loss_weight*LossFunc(class_logits,Y) + (1-loss_weight)*loss

          loss.backward()
          optimizer.step()

          running_loss+=loss.item() # sum to caluclate average loss per sample later
        
        val_acc,val_ice = metrics(cls, selector,k,M,N,iter_num,valloader,imgs_with_random_patch_val,selector,intrinsic=True)
        val_accs.append(val_acc)
        val_ices.append(val_ice)
        model_checkpoint = os.path.join(checkpoint_path,dataset_name+str(iter_num)+'_'+str(epoch)+'_Interpretable_selector.pt')
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
      best_model_path = os.path.join(checkpoint_path,dataset_name+str(iter_num)+'_'+str(best_epoch)+'_Interpretable_selector.pt')
      best_model = modifiedViT(
              image_size = 28,
              patch_size = 4,
              num_classes = num_classes,
              channels = 1,
              dim = 128,
              depth = 2,
              heads = 4,
              mlp_dim = 256,
              dropout = 0.1,
              emb_dropout = 0.1,
              explain = True).to(device)
      checkpoint = torch.load(best_model_path)
      best_model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      test_acc,test_ice = metrics(cls, best_model,k,M,N,iter_num,testloader,imgs_with_random_patch_test,best_model,intrinsic=True)
      
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
    parser.add_argument('--dataset_name',  type=str,help="Dataset type: Options:[fmnist, mnist]", default= 'mnist')
    parser.add_argument('--loss_weight',  type=str,help="weight assigned to selection loss", default= "0.9")
    parser.add_argument('--num_patches',  type=str,help="number of patches to select: Options[2,4,6,8,10]", default= "10")
    parser.add_argument('--validation', type=str,help=" Perform validation on validation or test set: Options:[without_test, with_test]",default="with_test")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    num_patches = int(args.num_patches)
    loss_weight = float(args.loss_weight)
    validation = args.validation  
    train_eval(dataset_name,loss_weight,num_patches,validation)