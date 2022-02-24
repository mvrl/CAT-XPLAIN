import torchtext
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe
import string
from torchtext.data.utils import get_tokenizer
from config import num_init
from text_models import TextTransformer, modifiedTextTransformer
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from config import *



def datapoints_counter(data_iter):
  text_count = 0 
  for i in range(len(data_iter)):
    item = next(iter(data_iter))
    for j in range(item.shape[0]):
      text_count+=1 # updating texr entry count

  return text_count



def text_with_random_sentence_generator(valloader,no_datapoints,max_length,num_sents):

    num_validation_texts = no_datapoints
    texts_with_random_sentence = np.zeros((num_init,num_validation_texts,max_length))
    text_count = 0
    for i in range(num_init):
      for item, label in valloader:
        one_batch = item
        for j in range(len(one_batch)):
          patch_selection_map = np.zeros(max_length)
          patch_selection_map[:num_sents] = 1
          texts_with_random_sentence[i][text_count] = np.random.shuffle(patch_selection_map)
          text_count+=1 # updating image count
      text_count = 0   

    return texts_with_random_sentence

def random_mask_generator(max_length,num_sents):
  patch_selection_map = np.zeros(max_length)
  patch_selection_map[:num_sents] = 1
  np.random.shuffle(patch_selection_map)

  return patch_selection_map

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


def generate_xs_text(X,selector,k,intrinsic=False):
    batch_size = X.shape[0]
    X = X.to(device)
  # 1: get the logits from the selector for instance X
    with torch.no_grad():
        if intrinsic:
            logits = selector.forward(X)[1]
        else:
            logits = selector.forward(X) # shape is (bs,max_length), where max_length is max number of sents in sentence
  # 2: get a subset of the features(encoded in a binary vector) by using the gumbel-softmax trick
    selected_subset = sample_concrete(tau,k,logits,train=False)# get S_bar from explainer
  # 3: get S from S_bar
    selected_subset_inverted = torch.abs(selected_subset-1)# getting S from S_bar
    return selected_subset_inverted


'''
custom loss function that is for our objective function(similar to categorical cross entropy function)
p_y_xs is the p(y|xs) or f_{bb}(xs)
p_y_xs is the p(y|x) or f_{bb}(x)
'''
def custom_loss(p_y_xs,p_y_x,batch_size):
    loss= torch.mean(torch.sum(p_y_x.view(batch_size, -1) * torch.log(eps+p_y_xs.view(batch_size, -1)), dim=1))
    return loss



def metrics(selector,k,init_num,valloader,bb_model,max_length,num_sents,intrinsic=False):  
  # if intrinsic:
  #   bb_model = selector
  correct_count, all_count = 0, 0
  ICE = 0
  for item, label in valloader:
    one_batch = item
    batch_label = label
    for i in range(len(one_batch)):
      text = one_batch[i].unsqueeze(0).to(device)
      label = batch_label[i].to(device)
      selected_subset = generate_xs_text(text,selector,k,intrinsic=intrinsic).to(device)
      # xprime_subset = torch.tensor(texts_with_random_val[init_num][all_count]).unsqueeze(0).to(device)
      xprime_subset = torch.tensor(random_mask_generator(max_length,num_sents)).unsqueeze(0).to(device)
      with torch.no_grad():
    # get the augmented image(from val. dataset)
        if intrinsic: #Interpretable Transformer
            out_xs = F.softmax(bb_model(text,selected_subset)[0],dim=1)
            out_xprime = F.softmax(bb_model(text, xprime_subset)[0],dim=1)
            out_x = F.softmax(bb_model(text)[0],dim=1)
        else: # Normal Transformer
            out_xs = F.softmax(bb_model(text,selected_subset),dim=1)
            out_xprime = F.softmax(bb_model(text, xprime_subset),dim=1)
            out_x = F.softmax(bb_model(text),dim=1)
      pred_label = torch.argmax(out_xs)
      true_label = torch.argmax(out_x)
      true_label = label
# post hoc accuracy calc.
      if(true_label == pred_label):
          correct_count += 1
    #ICE calc.    
      ICE+=out_xs[0][true_label]-out_xprime[0][true_label]
    #ICE+=out_xs[0][labels[i]]-out_xprime[0][labels[i]]
    #ICE+= F.kl_div(out_xs[0].log(), out_xprime[0]).item()
      all_count += 1

  ph_acc = (correct_count/all_count)
  ACE=ICE/all_count   

  return ph_acc,ACE.cpu()



def train_basemodel(data_type,trainloader,valloader,bb_model,LossFunc,optimizer,num_epochs,batch_size,checkpoint_path):
  train_loss = []
  valid_loss = []
  avg_train_losses = []
  avg_valid_losses = []
  #training loop
  for epoch in range(num_epochs):
    bb_model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
      for item, label in tepoch:
        data = item.to(device)
        target = label.long().to(device)
        
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
      for item, label in vtepoch:
        data = item.to(device)
        target = label.long().to(device)
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
            },checkpoint_path+'/'+data_type+'_model.pt')

  return bb_model 



def test_basemodel(valloader,bb_model):
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for item, label in valloader:
    data = item.to(device)
    labels = label.long().to(device)
    for i in range(len(labels)):
      text = data[i].to(device)
      text = text.unsqueeze(0)
      with torch.no_grad():
          out = bb_model(text)

      pred_label = torch.argmax(out)
      true_label = labels[i].to(device)
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("Number Of text reviews Tested =", all_count)
  print("Model Accuracy =", (correct_count/all_count)) 


def initialize_model(model_type,emb_dim,dim,depth,num_classes,max_length,device,train_emb):
    if model_type == 'transformer':
        model = TextTransformer(num_classes =num_classes, max_length=max_length, emb_dim=emb_dim,dim=dim, 
                depth=depth, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = 64, dropout = 0.,
                emb_dropout = 0.,train_emb = True).to(device)
    if model_type == 'exptransformer':
        model = modifiedTextTransformer(num_classes = num_classes, max_length=max_length, emb_dim = emb_dim,
                dim=dim, depth=depth, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = 64,
                dropout = 0., emb_dropout = 0.,explain=True,train_emb = True).to(device)
    return model#.double()



