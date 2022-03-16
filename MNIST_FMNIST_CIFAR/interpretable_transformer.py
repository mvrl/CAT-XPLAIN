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
from models import modifiedViT, initialize_model
from utils import sample_concrete, custom_loss, generate_xs, metrics, imgs_with_random_patch_generator, load_dataset, train_post_expmodel
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

val_true_acc_list = []
val_acc_list = []
val_ice_list = []

test_true_acc_list = []
test_acc_list = []
test_ice_list = []


def train_eval(dataset_name,dataset_class,loss_weight,depth,dim,num_patches,validation):
  seed_initialize(seed = 12345)
  if dataset_class == 'full':
      num_classes = 10
  else:
      num_classes = 2
  ###################################### LOAD DATASET ######################################################
  cls, trainloader, valloader, testloader, train_datasize, valid_datasize, test_datasize = load_dataset(dataset_name=dataset_name,dataset_class=dataset_class)

  if dataset_name == 'cifar':
    input_dim = 32
    channels = 3
    # M*N x M*N is the size of the image
    M = 8 # selection map size(assuming a square shaped selection map) 
    N = 4 # patch size(square patch)
  else:
    input_dim = 28
    channels = 1
    # M*N x M*N is the size of the image
    M = 7 # selection map size(assuming a square shaped selection map) 
    N = 4 # patch size(square patch)
  
  num_patches = int(num_patches*M*M)
  k = M*M-int(num_patches)# number of patches for S_bar
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
      model_type = 'expViT'
      bb_model = initialize_model(model_type,num_classes=num_classes,input_dim=input_dim, channels=channels,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device)
     
      selector = bb_model
      LossFunc = torch.nn.CrossEntropyLoss(size_average = True)
      #optimizer
      optimizer = torch.optim.Adam(selector.parameters(),lr = lr)
      # variable for keeping track of best ph_acc across different iterations 
      val_true_accs = []
      val_accs = []
      val_ices = []

      # training loop
      for epoch in range(num_epochs):  
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
        # get the inputs
          X, Y = data
          X = X.to(device)
          if num_classes != 2:
            Y = Y.long().to(device)
          else:
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
        
        val_acc,val_ice,_,_, val_true_acc = metrics(cls, selector,k,M,N,iter_num,valloader,imgs_with_random_patch_val,selector,intrinsic=True)
        val_true_accs.append(val_true_acc)
        val_accs.append(val_acc)
        val_ices.append(val_ice)
        if not os.path.exists(checkpoint_path):
          os.makedirs(checkpoint_path)
        model_checkpoint = os.path.join(checkpoint_path,dataset_name+'_'+str(num_classes)+'_'+str(iter_num)+'_'+str(epoch)+'_Interpretable_selector.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': selector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_checkpoint)     
          # print loss and validation accuracy at the end of each epoch 
        print('\nInitialization Number %d-> epoch: %d, average loss: %.3f, val_acc: %.3f, ICE: %.3f \n' %(iter_num+1, epoch + 1,running_loss/i, val_acc, val_ice))

      if all_metrics == True:
        best_val_performance = [(val_accs[item]+val_ices[item]+val_true_accs[item])/3 for item in range(len(val_ices))]
      else:
        best_val_performance = [(val_accs[item]+val_ices[item])/2 for item in range(len(val_ices))]
      best_epoch = np.argmax(best_val_performance)
      val_acc_list.append(val_accs[best_epoch])
      val_ice_list.append(val_ices[best_epoch])
      val_true_acc_list.append(val_true_accs[best_epoch])

      print("BEST EPOCH BASED ON VAL PERFORMANCE:",best_epoch)
      print("BEST (VAL_ACC,VAL_ICE)",(val_accs[best_epoch],val_ices[best_epoch]))
      best_model_path = os.path.join(checkpoint_path,dataset_name+'_'+str(num_classes)+'_'+str(iter_num)+'_'+str(best_epoch)+'_Interpretable_selector.pt')
      best_model = initialize_model(model_type,num_classes=num_classes,input_dim=input_dim, channels=channels,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device)
     
      checkpoint = torch.load(best_model_path)
      best_model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      test_acc,test_ice,_,_,test_true_acc = metrics(cls, best_model,k,M,N,iter_num,testloader,imgs_with_random_patch_test,best_model,intrinsic=True)
      
      ## Now train the model for 5 more epochs without the explainable component
      if train_further == True:
        test_acc = train_post_expmodel(iter_num,LossFunc,model_type,best_model,input_dim,channels,dim,N, M, depth,post_num_epochs,tau,k,batch_size,num_classes,cls,trainloader,valloader,testloader,optimizer,checkpoint_path,dataset_name,device)
      
      if validation == 'with_test':
        print('test (ph acc, ICE, ACC):',(test_acc,test_ice, test_true_acc))

      test_true_acc_list.append(test_true_acc)
      test_acc_list.append(test_acc)
      test_ice_list.append(test_ice)

        
  print('mean val ph acc: %.3f'%(np.mean(val_acc_list)),', std dev: %.3f '%(np.std(val_acc_list))) 
  print('mean val ICE: %.3f'%(np.mean(val_ice_list)),', std dev: %.3f '%(np.std(val_ice_list))) 
  print('mean val true acc with whole input: %.3f'%(np.mean(val_true_acc_list)),', std dev: %.3f '%(np.std(val_true_acc_list))) 

  if validation == 'with_test':
    print('mean test ph acc: %.3f'%(np.mean(test_acc_list)),', std dev: %.3f '%(np.std(test_acc_list))) 
    print('mean test ICE: %.3f'%(np.mean(test_ice_list)),', std dev: %.3f '%(np.std(test_ice_list)))
    print('mean test true acc with whole input: %.3f'%(np.mean(test_true_acc_list)),', std dev: %.3f '%(np.std(test_true_acc_list)))  

  print('\nDONE! \n')


if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',  type=str,help="Dataset type: Options:[fmnist, mnist, cifar]", default= 'mnist')
    parser.add_argument('--depth',  type=str,help="depth of the transformer block: Options[1,2,4,8,10]", default= "2")
    parser.add_argument('--dim',  type=str,help="dimension of hidden state: Options[64,128,256,512]", default= "128")
    parser.add_argument('--loss_weight',  type=str,help="weight assigned to selection loss", default= "0.60")
    parser.add_argument('--num_patches',  type=str,help="frac for number of patches to select: Options[0.05,0.10,0.25,0.50,0.75]", default= "0.25")
    parser.add_argument('--validation', type=str,help=" Perform validation on validation or test set: Options:[without_test, with_test]",default="with_test")
    parser.add_argument('--dataset_class', type=str,help="select_model type: Options:[partial,full]",default="partial")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    num_patches = float(args.num_patches)
    loss_weight = float(args.loss_weight)
    validation = args.validation 
    depth = int(args.depth)
    dim = int(args.dim)
    dataset_class = args.dataset_class 

    train_eval(dataset_name=dataset_name,dataset_class=dataset_class,loss_weight=loss_weight,
            depth=depth,dim=dim,num_patches=num_patches,validation=validation)