# ## Most of the code below is taken from: https://github.com/pranoy-panda/Causal-Feature-Subset-Selection


from models import TextTransformer
import itertools
from easydict import EasyDict as edict

# relevant libraries
import numpy as np
import torch
import torch.utils.data as data
from time import time
from torch import nn, optim
import torch.nn.functional as F
import random
from joblib import dump, load
from tqdm import tqdm
from text_utils import num2words, get_imdb, text_with_random_word_generator, random_mask_generator, sample_concrete, generate_xs_text, metrics
from text_utils import train_basemodel, test_basemodel, initialize_model, custom_loss
from config import *
import os

# for supressing warnings
import warnings
warnings.filterwarnings('ignore')


# set the seeds for reproducibility
def seed_initialize(seed = 12345):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  SEED = seed
  random.seed(SEED) 
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)

def train_eval(dataset_name, bb_model_type, sel_model_type, num_words,validation='without_test'):
  seed_initialize(seed = 12345)
  trainloader, traincount, valloader, validcount, testloader, testcount, vectors, vocab = get_imdb(batch_size=batch_size, max_length=max_length,device=device)
  print("For dataset:",dataset_name)
  print("For Experiment with bb_model:",bb_model_type)
  print("For Experiment with sel_model:",sel_model_type)
  print('1. Training the Basemodel...... \n')

  num_words = int(num_words*max_length)
  k = max_length - int(num_words)# number of words for S_bar
  
  ## Initialize Base model
  bb_model = initialize_model(model_type=bb_model_type,vocab_emb=vectors,num_classes=num_classes,max_length=max_length,emb_dim=emb_dim,device=device)

  LossFunc_basemodel = torch.nn.CrossEntropyLoss(size_average = True)
  optimizer_basemodel = torch.optim.Adam(bb_model.parameters(),lr = lr_basemodel) 

  # train the basemodel 
  bb_model = train_basemodel(trainloader,valloader,bb_model,
            LossFunc_basemodel,optimizer_basemodel,,num_epochs,batch_size,checkpoint_path)

  # testing the model on held-out validation dataset
  if validation == 'without_test':
    test_basemodel(valloader,bb_model)
  else:
    test_basemodel(testloader,bb_model)
  print('Basemodel trained! \n')

  ##############################################################

  print('2. Starting main feature selection algorithm......... \n')

  # '''
  # generate sentences with random words selected for calculating the ACE metric
  # '''
  # texts_with_random_word_val = text_with_random_word_generator(valloader,validcount,max_length,num_words)
  # texts_with_random_word_test = text_with_random_word_generator(testloader,testcount,max_length,num_words)

  # training loop where we run the experiments for multiple times and report the 
# mean and standard deviation of the metrics ph_acc and ICE.
  test_acc_list = []
  test_ice_list = []

  val_acc_list = []
  val_ice_list = []

  for iter_num in range(num_init):
    # intantiating the gumbel_selector or in other words initializing the explainer's weights
    ## Initialize Selection model
    selector = initialize_model(model_type=sel_model_type,vocab_emb=vectors,num_classes=max_length,max_length=max_length,emb_dim=emb_dim,device=device)
    #optimizer
    optimizer = torch.optim.Adam(selector.parameters(),lr = lr)
    
    val_accs = []
    val_ices = []
    # training loop
    for epoch in range(num_epochs):  
        running_loss = 0
        for item in enumerate(trainloader, 0):
            X = item.text[0].to(device)
            Y = item.label.long().to(device)
            batch_size = X.size(0)
            # zero the parameter gradients
            optimizer.zero_grad()
            # 1: get the logits from construct_gumbel_selector()
            logits = selector.forward(X)
            # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick      
            selected_subset = sample_concrete(tau,k,logits,train=True) # get S_bar from explainer
            f_xsbar = F.softmax(bb_model(X,selected_subset)) # f_xs stores p(y|xs)
            with torch.no_grad():
              f_x =  F.softmax(bb_model(X)) # f_x stores p(y|x)          

            # optimization function
            loss = custom_loss(f_xsbar,f_x,batch_size)

            loss.backward()
            optimizer.step()
          
            running_loss+=loss.item() # average loss per sample
  ################################################################################
        val_acc,val_ice = metrics(selector,k,init_num,valloader,bb_model,max_length,num_words,intrinsic=False)
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
    best_model = initialize_model(model_type=sel_model_type,vocab_emb=vectors,num_classes=max_length,max_length=max_length,emb_dim=emb_dim,device=device)
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    ## Initialize base blackbox model
    bb_checkpoint = torch.load(checkpoint_path+'_model.pt')
    bb_model = initialize_model(model_type=bb_model_type,vocab_emb=vectors,num_classes=num_classes,max_length=max_length,emb_dim=emb_dim,device=device)
    bb_model.load_state_dict(bb_checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_acc,test_ice = metrics(best_model,k,iter_num,valloader,bb_model,max_length,num_words,intrinsic=False)
    
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
    parser.add_argument('--dataset_name',  type=str,help="Dataset type: Options:[imdb]", default= 'imdb')
    parser.add_argument('--bb_model_type', type=str,help="Base_model type: Options:[transformer]",default="transformer")
    parser.add_argument('--sel_model_type', type=str,help="select_model type: Options:[transformer]",default="transformer")
    parser.add_argument('--num_words',  type=str,help="frac for number of words to select: Options[0.05,0.10,0.25,0.50,0.75]", default= "0.25")
    parser.add_argument('--validation', type=str,help=" Perform validation on validation or test set: Options:[without_test, with_test]",default="with_test")
    parser.add_argument('--sweep', type=str,help="select_model type: Options:[sweep,no_sweep]",default="no_sweep")
    args = parser.parse_args()

    validation = args.validation
    num_words = float(args.num_words)
    HyperParameters = edict()
    HyperParameters.dataset_name = ["imdb"]
    HyperParameters.bb_model_type = ["transformer"]
    HyperParameters.sel_model_type = ["transformer"]
    HyperParameters.params = [HyperParameters.dataset_name, HyperParameters.bb_model_type,HyperParameters.sel_model_type]
    params = list(itertools.product(*HyperParameters.params))
    if args.sweep == "sweep":
      for hp in params:
        dataset_name = hp[0]
        bb_model_type = hp[1]
        sel_model_type = hp[2]
        print("For parameters:(dataset_name, BaseModel,Selector)",hp)
        train_eval(dataset_name, bb_model_type, sel_model_type,num_words,validation)
    else:
      dataset_name = args.dataset_name
      bb_model_type = args.bb_model_type
      sel_model_type = args.sel_model_type
      train_eval(dataset_name, bb_model_type, sel_model_type,num_words,validation)



