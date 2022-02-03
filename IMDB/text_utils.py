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


def num2words(vocab, vec):
    """
        Converts a vector of word indicies
        to a list of strings
    """
    return [vocab.itos[i] for i in vec]

def datapoints_counter(data_iter):
  text_count = 0 
  for i in range(len(data_iter)):
    item = next(iter(data_iter))
    for j in range(item.text[0].shape[0]):
      text_count+=1 # updating texr entry count

  return text_count


def get_imdb(batch_size=64, max_length=250,emb_dim=30,device='cpu'):
# Adapted from from https://github.com/PrideLee/sentiment-analysis/blob/6637f8f9dfb44308ca47a3e92fcdd7638e62a485/transformer/dataloader.py
  tokenizer = get_tokenizer('basic_english')
  TEXT = torchtext.legacy.data.Field(lower=True, include_lengths=True, batch_first=True, tokenize=tokenizer, fix_length=max_length)
  LABEL = torchtext.legacy.data.Field(sequential=False, unk_token=None, pad_token=None)
  train_set, test_set = torchtext.legacy.datasets.IMDB.splits(TEXT, LABEL)
  train_set, valid_set = torchtext.legacy.data.Dataset.split(train_set,split_ratio=0.85, stratified=False, strata_field='label', random_state=None)

  TEXT.build_vocab(train_set, vectors=GloVe(name='6B', dim=emb_dim, max_vectors=500000))
  LABEL.build_vocab(train_set)

  # print vocab information
  print('len(TEXT.vocab)', len(TEXT.vocab))
  print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

  # make iterator for splits based on the batch_size
  train_iter, valid_iter, test_iter = torchtext.legacy.data.BucketIterator.splits((train_set, valid_set, test_set), batch_size=batch_size, device=device)
  # Print number of batches
  print("Number of texts in each data split(train/val/test):")
  traincount = datapoints_counter(train_iter)
  validcount = datapoints_counter(valid_iter)
  testcount = datapoints_counter(test_iter)
  print(traincount,validcount,testcount)

  return train_iter, traincount, valid_iter, validcount, test_iter, testcount, TEXT.vocab.vectors, TEXT.vocab

def text_with_random_word_generator(valloader,no_datapoints,max_length,num_words):

    num_validation_texts = no_datapoints
    texts_with_random_word = np.zeros((num_init,num_validation_texts,max_length))
    text_count = 0
    for i in range(num_init):
      for item in valloader:
        one_batch = item.text[0]
        for j in range(len(one_batch)):
          patch_selection_map = np.zeros(max_length)
          patch_selection_map[:num_words] = 1
          texts_with_random_word[i][text_count] = np.random.shuffle(patch_selection_map)
          text_count+=1 # updating image count
      text_count = 0   

    return texts_with_random_word

def random_mask_generator(max_length,num_words):
  patch_selection_map = np.zeros(max_length)
  patch_selection_map[:num_words] = 1
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
            logits = selector.forward(X) # shape is (bs,max_length), where max_length is max number of words in sentence
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



def metrics(selector,k,init_num,valloader,bb_model,max_length,num_words,intrinsic=False):  
  # if intrinsic:
  #   bb_model = selector
  correct_count, all_count = 0, 0
  ICE = 0
  for item in valloader:
    one_batch = item.text[0]
    batch_label = item.label
    for i in range(len(one_batch)):
      text = one_batch[i].unsqueeze(0).to(device)
      label = batch_label[i].to(device)
      selected_subset = generate_xs_text(text,selector,k,intrinsic=intrinsic).to(device)
      # xprime_subset = torch.tensor(texts_with_random_val[init_num][all_count]).unsqueeze(0).to(device)
      xprime_subset = torch.tensor(random_mask_generator(max_length,num_words)).unsqueeze(0).to(device)
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



def train_basemodel(trainloader,valloader,bb_model,LossFunc,optimizer,num_epochs,batch_size,checkpoint_path):
  train_loss = []
  valid_loss = []
  avg_train_losses = []
  avg_valid_losses = []
  #training loop
  for epoch in range(num_epochs):
    bb_model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
      for item in tepoch:
        data = item.text[0].to(device)
        target = item.label.long().to(device)
        
        tepoch.set_description("Epoch "+str(epoch))
        
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = bb_model(data)
        loss = LossFunc(outputs, target)
        train_loss.append(loss.item())

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == target).sum().item()
        accuracy = correct / batch_size
        
        loss.backward()
        optimizer.step()

        tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    
    bb_model.eval()
    with tqdm(valloader, unit="batch") as vtepoch:
      for item in vtepoch:
        data = item.text[0].to(device)
        target = item.label.long().to(device)
        vtepoch.set_description("Epoch "+str(epoch))
        outputs = bb_model(data)
        valloss = LossFunc(outputs, target)
        valid_loss.append(valloss.item())
        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == target).sum().item()
        accuracy = correct / batch_size
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
            },checkpoint_path+'_model.pt')

  return bb_model 



def test_basemodel(valloader,bb_model):
  # testing the black box model performance on the entire validation dataset
  correct_count, all_count = 0, 0
  for item in valloader:
    data = item.text[0].to(device)
    labels = item.label.long().to(device)
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

  print("Number Of Images Tested =", all_count)
  print("Model Accuracy =", (correct_count/all_count)) 


def initialize_model(model_type,vocab_emb,emb_dim,dim_head,depth,num_classes,max_length,device):
    if model_type == 'transformer':
        model = TextTransformer(vocab_emb=vocab_emb, num_classes =num_classes, max_length=max_length, dim=emb_dim, 
                depth=depth, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = dim_head, dropout = 0.,
                emb_dropout = 0.,train_word_embeddings=True).to(device)
    if model_type == 'exptransformer':
        model = modifiedTextTransformer(vocab_emb=vocab_emb, num_classes = num_classes, max_length=max_length,
                dim=emb_dim, depth=depth, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = dim_head,
                dropout = 0., emb_dropout = 0.,explain=True,train_word_embeddings=True).to(device)
    return model.double()



