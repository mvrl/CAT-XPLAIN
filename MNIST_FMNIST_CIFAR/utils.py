## Most of the code below is taken from: https://github.com/pranoy-panda/Causal-Feature-Subset-Selection

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
import torch.utils.data as data
from torchvision import datasets as vision_datasets
from torchtext import datasets as text_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from config import *
from tqdm import tqdm
from models import modifiedViT, initialize_model
#################################################################################################################################


######################                           DATA LOADING                #########################################


################################################################################################################################
def mask_generator(dataset,cls):
  mask = []
  for i in range(len(dataset)):
    if dataset[i][1] in cls:
      mask.append(True)
    else:
      mask.append(False)
  
  return mask, sum(mask)

def load_dataset(dataset_name='mnist',dataset_class="partial"):

  if dataset_name=='fmnist':
    train_set = vision_datasets.FashionMNIST(root=data_path, transform=ToTensor(), download=True, train=True)
    test_set = vision_datasets.FashionMNIST(root=data_path, transform=ToTensor(), download=True, train=False)
    if dataset_class == 'partial':
      cls = [0, 9]
    else:
      cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  if dataset_name == 'mnist':
    train_set = vision_datasets.MNIST(root=data_path, transform=ToTensor(), download=True, train=True)
    test_set = vision_datasets.MNIST(root=data_path, transform=ToTensor(), download=True, train=False)
    if dataset_class == 'partial':
      cls = [3, 8]
    else:
      cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  
  if dataset_name == 'cifar':
    train_set = vision_datasets.CIFAR10(root=data_path, transform=ToTensor(), download=True, train=True)
    test_set = vision_datasets.CIFAR10(root=data_path, transform=ToTensor(), download=True, train=False)
    if dataset_class == 'partial':
      cls = [2, 9]
    else:
      cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  if dataset_name == 'imdb':
    train_set = text_datasets.IMDB(root=data_path, transform=ToTensor(), split='train')
    test_set = text_datasets.IMDB(root=data_path, transform=ToTensor(), split='test')
    cls = [0, 1]
    

  # Before
  print('Train data set:', len(train_set))
  print('Test data set:', len(test_set))

  # Random split
  train_set_size = int(len(train_set) * 0.8)
  valid_set_size = len(train_set) - train_set_size
  train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])

  # After
  print('='*30)
  print('Train data set:', len(train_set))
  print('Valid data set:', len(valid_set))
  print('Test data set:', len(test_set))

  trainloader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                            sampler = SubsetRandomSampler(np.where(mask_generator(train_set,cls=cls)[0])[0]), num_workers=2) #to simulate Shuffle
  valloader = torch.utils.data.DataLoader(valid_set, batch_size=64,
                                            sampler = np.where(mask_generator(valid_set,cls=cls)[0])[0], num_workers=2)
  testloader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                            sampler = np.where(mask_generator(test_set,cls=cls)[0])[0], num_workers=2)

  train_datasize = mask_generator(train_set,cls)[1]
  valid_datasize = mask_generator(valid_set,cls)[1]
  test_datasize = mask_generator(test_set,cls)[1]
  print(train_datasize,valid_datasize,test_datasize)
  print(len(trainloader),len(valloader), len(testloader))
  print("classes:",cls)

  return cls, trainloader, valloader, testloader, train_datasize, valid_datasize, test_datasize

##################### RANDOM PATCH GENERATOR FOR VALIDATION AND TEST DATASET #############################
def imgs_with_random_patch_generator(valloader,no_datapoints,num_patches):
  '''
  uncomment to generate images with random patches selected
  '''
  num_validation_images = no_datapoints
  img_size =  next(iter(valloader))[0].shape[-1]
  channels = next(iter(valloader))[0].shape[1]
  imgs_with_random_patch = np.zeros((num_init,num_validation_images,channels,img_size,img_size))
  img_count = 0
  M_dict = {1:7,3:8} #if channels is 1 make M 7 else 8 for cifar
  M = M_dict[channels]
  for i in range(num_init):
    for images,_ in valloader:
      for j in range(len(images)):
        patch_selection_map = np.zeros((M*M))
        patch_selection_map[:num_patches] = 1
        np.random.shuffle(patch_selection_map) # random permutation of the above created array with 'num_patches' ones
        patch_selection_map = patch_selection_map.reshape(M,M)
        patch_selection_map = np.kron(patch_selection_map, np.ones((N,N))) # upsampled to size (M*N,M*N)
        if channels != 1:
          patch_selection_map = np.tile(patch_selection_map,(channels,1,1))  
        imgs_with_random_patch[i][img_count] = np.multiply(patch_selection_map,images[j])
        img_count+=1 # updating image count
    img_count = 0    
  return imgs_with_random_patch

######################################################################################################

# This cell implements the real code i.e. where the explainer(gumbel_selector()) is trained

'''
This function samples from a concrete distribution during training and while inference, it gives the indices of the top k logits
'''
def sample_concrete(tau,k,logits,train=True):
  # input logits dimension: [batch_size,1,d]
  logits = logits.unsqueeze(1)
  d = logits.shape[2]
  batch_size = logits.shape[0]  
  if train == True:
    softmax = nn.Softmax().to(device) # defining the softmax operator
    unif_shape = [batch_size,k,d] # shape for uniform distribution, notice there is k. Reason: we have to sample k times for k features
    uniform = (1 - 0) * torch.rand(unif_shape).to(device) # generating vector of shape "unif_shape", uniformly random numbers in the interval [0,1)
    gumbel = - torch.log(-torch.log(uniform)) # generating gumbel noise/variables
    noisy_logits = (gumbel + logits)/tau # perturbed logits(perturbed by gumbel noise and temperature coeff. tau)
    samples = softmax(noisy_logits) # sampling from softmax 
    samples,_ = torch.max(samples, axis = 1)
    
    return samples
  else:  
    logits = torch.reshape(logits,[-1, d]) 
    discrete_logits = torch.zeros(logits.shape[1])
    vals,ind = torch.topk(logits,k)
    discrete_logits[ind[0]]=1    
    discrete_logits = discrete_logits.type(torch.float32) # change type to float32
    discrete_logits = torch.unsqueeze(discrete_logits,dim=0)
    return discrete_logits 

##############################################################################################################################
'''
custom loss function that is for our objective function(similar to categorical cross entropy function)
p_y_xs is the p(y|xs) or f_{bb}(xs)
p_y_xs is the p(y|x) or f_{bb}(x)
'''
def custom_loss(p_y_xs,p_y_x,batch_size):
    loss= torch.mean(torch.sum(p_y_x.view(batch_size, -1) * torch.log(eps+p_y_xs.view(batch_size, -1)), dim=1))
    return loss

###############################################################################################################################
'''
Given an instance X and the selector network, this function returns X_S,X_Sbar and S_bar
X_S: augmented X, where the un-selected patches are masked out(here, replaced by zero) 
X_Sbar: augmented X, where the selected patches are masked out(here, replaced by zero)
S_bar: a map/2D matrix where the un-selected patches are set to 1 and selected patches are set to 0
'''
def generate_xs(X,selector,k,M,N,intrinsic=False): # M x M is the size of the patch, and M*N x M*N is the size of the instance X
    batch_size = X.shape[0]
    channels = X.shape[1]
    X = X.to(device)
  # 1: get the logits from the selector for instance X
    with torch.no_grad():
        if intrinsic:
            logits = selector.forward(X)[1]
        else:
            logits = selector.forward(X) # shape is (bs,M*M), where M is the patch size
  # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick
    selected_subset = sample_concrete(tau,k,logits,train=False)# get S_bar from explainer
  # 3: reshape selected_subset to the size M x M i.e. the size selection map
    selected_subset = torch.reshape(selected_subset,(batch_size,M,M))
    selected_subset = torch.unsqueeze(selected_subset,dim=1)# S_bar
    selected_subset_inverted = torch.abs(selected_subset-1)# getting S from S_bar
  # 4: upsample the selection map
    upsample_op = nn.Upsample(scale_factor=N, mode='nearest')
    v = upsample_op(selected_subset_inverted).to(device)
  # 5: X_S = elementwise_multiply(X,v); compute f_{bb}(X_S)
    X_S = torch.mul(X,v) # output shape will be [bs,1,M*N,M*N] 
  #X_Sbar = torch.mul(X,v_bar)
    return X_S,v#,X_Sbar


#######################################################################################################################
'''
This function calculates thes two metrics for evaluating the explanations
1. post hoc accuracy
2. average ICE: (1/batch_size)*( p(y=c/xs) - p(y=c/x') ), here c is class 
   predicted by basemodel and x' is the image where k patches are randomly selected from x are present, rest all patches are null
'''
def metrics(cls,selector,k,M,N,init_num,valloader,imgs_with_random_patch,bb_model,intrinsic=False):
  
  num_classes = len(cls)
  # if intrinsic:
  #   bb_model = selector
  correct_count, all_count, ph_correct_count, true_correct_count = 0, 0, 0, 0
  ICE, ICE_ph = 0,0
  for images,labels in valloader:
    for i in range(len(images)):
      img = images[i].unsqueeze(0).to(device)
      if num_classes != 2:
        label = labels[i].to(device)
      else:
        label = torch.tensor([1 if labels[i] == cls[-1] else 0][0]).to(device)
      xs,v = generate_xs(img,selector,k,M,N,intrinsic)
      xprime = torch.Tensor(imgs_with_random_patch[init_num][all_count]).unsqueeze(0).to(device)
      with torch.no_grad():
    # get the augmented image(from val. dataset)
        if intrinsic: #Interpretable Transformer
            out_xs = F.softmax(bb_model(xs)[0],dim=1)
            out_xprime = F.softmax(bb_model(xprime)[0],dim=1)
            out_x = F.softmax(bb_model(img)[0],dim=1)
        else: # Normal Transformer
            out_xs = F.softmax(bb_model(xs),dim=1)
            out_xprime = F.softmax(bb_model(xprime),dim=1)
            out_x = F.softmax(bb_model(img),dim=1)
      pred_label = torch.argmax(out_xs)
      pred_label_full = torch.argmax(out_x)
      true_label = label
# post hoc accuracy calc.
      if(true_label == pred_label):
          correct_count += 1
      if(pred_label == pred_label_full):
          ph_correct_count += 1
      if(true_label == pred_label_full):
        true_correct_count += 1

    #ICE calc.    
      ICE +=out_xs[0][true_label]-out_xprime[0][true_label]
      ICE_ph +=out_xs[0][pred_label_full]-out_xprime[0][pred_label_full]
      all_count += 1

  acc = (correct_count/all_count)
  ph_acc = (ph_correct_count/all_count)
  overall_acc = (true_correct_count/all_count)
  
  ACE=ICE/all_count
  ACE_ph =  ICE_ph/all_count  

  return acc, ACE.cpu(), ph_acc, ACE_ph.cpu(), overall_acc


def train_basemodel(data_type,cls,trainloader,valloader,bb_model,LossFunc,optimizer,num_epochs,batch_size,checkpoint_path,depth,dim):
  train_loss = []
  valid_loss = []
  avg_train_losses = []
  avg_valid_losses = []
  num_classes = len(cls)
  #training loop
  for epoch in range(num_epochs):
    bb_model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
      for data, target in tepoch:
        data = data.to(device)
        if num_classes != 2:
          target = target.long().to(device)
        else:
          target = (target == cls[-1]).long().to(device)
        tepoch.set_description("Epoch "+str(epoch))
        
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = bb_model(data)
        loss = LossFunc(outputs, target)
        train_loss.append(loss.item())

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == target).sum().item()
        accuracy = correct / len(predictions)
        
        loss.backward()
        optimizer.step()

        tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    
    bb_model.eval()
    with tqdm(valloader, unit="batch") as vtepoch:
      for data, target in vtepoch:
        data = data.to(device)
        if num_classes != 2:
          target = target.long().to(device)
        else:
          target = (target == cls[-1]).long().to(device)
        vtepoch.set_description("Epoch "+str(epoch))
        outputs = bb_model(data)
        valloss = LossFunc(outputs, target)
        valid_loss.append(valloss.item())
        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == target).sum().item()
        accuracy = correct / len(predictions)
        vtepoch.set_postfix(loss=valloss.item(), accuracy=100. * accuracy)
    
    # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_loss)
        valid_loss = np.average(valid_loss)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs_basemodel))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs_basemodel:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        #print(print_msg)
        
        # clear lists to track next epoch
        train_loss = []
        valid_loss = []
        
  torch.save({
            'epoch': epoch,
            'model_state_dict': bb_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },checkpoint_path+'/'+data_type+'_class'+str(num_classes)+'_model_'+'depth_'+str(depth)+'_dim_'+str(dim)+'.pt')

  return bb_model 



def test_basemodel(cls,valloader,bb_model):
  num_classes = len(cls)
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for images,labels in valloader:
    images = images.to(device)
    if num_classes != 2:
      labels = labels.to(device)
    else:
      labels = (labels == cls[-1]).long().to(device)
    with torch.no_grad():
      out = bb_model(images)
      pred_label = torch.argmax(out,axis=1)
    correct_count = correct_count + sum(labels == torch.argmax(out,axis=1))
    all_count = all_count + len(labels)
  print("Number Of Images Tested =", all_count)
  print("Model Accuracy =", (correct_count/all_count))



def test_expmodel(cls,valloader,exp_model,device):
  num_classes = len(cls)
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for images,labels in valloader:
    images = images.to(device)
    if num_classes != 2:
      labels = labels.to(device)
    else:
      labels = (labels == cls[-1]).long().to(device)
    with torch.no_grad():
      out = exp_model(images)[0]
      pred_label = torch.argmax(out,axis=1)
    correct_count = correct_count + sum(labels == torch.argmax(out,axis=1))
    all_count = all_count + len(labels)
  
  acc =  correct_count/all_count
  return acc.cpu()

def train_prior_expmodel(iter_num,LossFunc,model_type,selector,input_dim,channels,dim,N,M,depth,pre_epochs,tau,k,batch_size,num_classes,cls,trainloader,valloader,testloader,optimizer,checkpoint_path,dataset_name,device):
# training loop
  val_accs = []
  for epoch in range(pre_epochs):  
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
      class_logits, patch_logits = selector.forward(X)   
      loss = LossFunc(class_logits,Y)

      loss.backward()
      optimizer.step()

      running_loss+=loss.item() # sum to caluclate average loss per sample later
    
    val_acc = test_expmodel(cls,valloader,selector,device)
    val_accs.append(val_acc)

    if not os.path.exists(checkpoint_path):
      os.makedirs(checkpoint_path)
    model_checkpoint = os.path.join(checkpoint_path,dataset_name+'_'+str(num_classes)+'_'+str(iter_num)+'_'+str(epoch)+'_Interpretable_selector_pre_train.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': selector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_checkpoint)     
      
  best_val_performance = val_accs
  best_epoch = np.argmax(best_val_performance)
  print("For CE pre-train BEST EPOCH BASED ON VAL PERFORMANCE:",best_epoch)
  print("For CE pre-train  BEST (VAL_ACC,VAL_ICE)",(val_accs[best_epoch]))
  best_model_path = os.path.join(checkpoint_path,dataset_name+'_'+str(num_classes)+'_'+str(iter_num)+'_'+str(best_epoch)+'_Interpretable_selector_pre_train.pt')
  best_model = initialize_model(model_type,num_classes=num_classes,input_dim=input_dim, channels=channels,patch_size=N,dim=dim,depth=depth,heads=8,mlp_dim=256,device=device)
  
  checkpoint = torch.load(best_model_path)
  best_model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return best_model, optimizer
    