#This has the functions related to data preparation and loading again from the same baseline code in Xin's repo
import timeit
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from config import imdb_data_path
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt',download_dir=imdb_data_path)
nltk.data.path.append(imdb_data_path)
sent_emb_model = SentenceTransformer(os.path.join(imdb_data_path,'all-MiniLM-L6-v2')) #dim 384
# All options can be seen in https://www.sbert.net/docs/pretrained_models.html
pad_sentence = "##### #####"
label_dict = {'neg':0,'pos':1}

def read_csv(data_file_path):
    data = list(pd.read_csv(os.path.join(imdb_data_path,data_file_path))['review'])
    label = list(pd.read_csv(os.path.join(imdb_data_path,data_file_path))['label'])
    return data, label

class Dataset_IMDB_sentence(Dataset):
    def __init__(self,
                 data_file,
                 num_sentences = 50, #Number of sentences to select
                 seed = 11
                 ):       
        self.reviews, self.labels = read_csv(data_file)
        self.seed = seed
        self.num_sentences = num_sentences 

    def __len__(self):
        return len(self.reviews)
    def __getitem__(self,idx):
        review = self.reviews[idx]
        label =  label_dict[self.labels[idx]]      
        sentences = sent_tokenize(review)
        if len(sentences) < 50:
            sentences = sentences + [pad_sentence]*(50-len(sentences))
        sentence_embeddings = sent_emb_model.encode(sentences)

        return torch.tensor(sentence_embeddings).double(), int(label)
