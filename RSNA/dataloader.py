#This has the functions related to data preparation and loading again from the same baseline code in Xin's repo
import timeit
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from config import csv_path, data_path
from pydicom import dcmread
import torch

def read_csv(data_file_path):
    ptids = list(pd.read_csv(os.path.join(csv_path,data_file_path))['patientId'])
    labels = list(pd.read_csv(os.path.join(csv_path,data_file_path))['Target'])
    return ptids, labels

class Dataset_RSNA(Dataset):
    def __init__(self,
                 data_file,
                 seed = 11
                 ):       
        self.ptids, self.labels = read_csv(data_file)
        self.seed = seed

    def __len__(self):
        return len(self.ptids)
    def __getitem__(self,idx):
        ptid = self.ptids[idx]
        label =  self.labels[idx]     
        dcm_img = dcmread(os.path.join(data_path,ptid+'.dcm'))
        dcm_array = dcm_img.pixel_array
        img = torch.tensor(dcm_array).unsqueeze(0)
        return img.float(), int(label)