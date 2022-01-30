#This has the functions related to data preparation and loading again from the same baseline code in Xin's repo
import timeit
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
import csv
from config import cfg

# Saved input Shape (1, 190, 190, 190)

#view_map = {'view0':'axial','view1':'sagittal','view2':'coronal'}

def read_csv(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data)
    return data

def random_patch_selector(img,M,N,num_patches):
    patch_selection_map = np.zeros((M*M))
    patch_selection_map[:num_patches] = 1
    np.random.shuffle(patch_selection_map) # random permutation of the above created array with 'num_patches' ones
    patch_selection_map = patch_selection_map.reshape(M,M)
    patch_selection_map = np.kron(patch_selection_map, np.ones((N,N))) # upsampled to size (M*N,M*N)   
    img_with_random_patch = np.multiply(patch_selection_map,img)

    return img_with_random_patch

def prep_tensor(im,random_patch,num_patches,M,N):
    if random_patch:
        im = random_patch_selector(im,M,N,num_patches)
    
    im = np.expand_dims(im, axis=0) # Creating a channel dim at axis =0 as required by pytorch
    
    return torch.tensor(im).float()
    

def prep_data(LABEL_PATH ,exper_path,TEST_NUM, groups):
    # This function is used to prepare train/test labels for 5-fold cross-validation
    TEST_LABEL = LABEL_PATH +groups+'/'+groups+ 'fold_' + str(TEST_NUM)+'.csv'

    # combine train labels
    filenames = [LABEL_PATH +groups +'/'+groups+'fold_0.csv', 
                LABEL_PATH +groups +'/'+groups+'fold_1.csv', 
                LABEL_PATH +groups +'/'+groups+'fold_2.csv', 
                LABEL_PATH +groups +'/'+groups+'fold_3.csv',
                LABEL_PATH +groups +'/'+groups+'fold_4.csv'
                ]

    filenames.remove(TEST_LABEL)

    trainlist = []
    for csvfile in filenames:
        df = pd.read_csv(csvfile,header=None)
        trainlist = trainlist + list(np.array(df).reshape(len(df)))
    
    traindf = pd.DataFrame(trainlist)
    TRAIN_LABEL = os.path.join(exper_path,str(TEST_NUM)+'_combined_train_list.csv')
    traindf.to_csv(TRAIN_LABEL,header=False,index=False)
            
    return TRAIN_LABEL, TEST_LABEL

class Dataset_MRI(Dataset):
    def __init__(self,
                 label_file,
                 groups,
                 view_type = '1', #Options: ['0' or '1' or '2' or 'multi']
                 precision = 'full',
                 random_patch=True, 
                 M = 19, #Number of patches in one axis of image
                 N = 10, #patch dimension
                 num_patches = 50, #Number of patches to select
                 seed = 11
                 ):       
        self.files = read_csv(label_file)
        self.groups = groups
        self.view_type = view_type
        self.precision = precision
        self.seed = seed
        self.random_patch = random_patch
        self.n_views = 3
        self.M = M
        self.N = N
        self.num_patches = num_patches

    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        temp = self.files[idx]        
        full_path = os.path.join(cfg.mri_data,temp[0])         
        #/mnt/gpfs2_16m/pscratch/nja224_uksr/xin_data/Preprocessed/AD/135_S_4863_brain.npy #Path
        label = full_path.split('/')[-2]
        numpy_file = full_path.split('/')[-1]
        imp_numpy_file = label+'_'+numpy_file 
        label1 = self.groups.split('_')[0]
        label2 = self.groups.split('_')[1]
        if(label==label1):
            label=0
        elif(label==label2):
            label=1
        else:
            print('Label Error')
        
        im = np.load(full_path)
        im = im[0,:,:,:]
        im0 = []
        im1 = []
        im2 = []

        middle_slice = int(im.shape[2]/2)
        if self.view_type == '0':
            im = im[middle_slice,:,:]
            im =  prep_tensor(im,self.random_patch,self.num_patches,self.M,self.N)
        if self.view_type == '1':
            im = im[:,middle_slice,:]
            im =  prep_tensor(im,self.random_patch,self.num_patches,self.M,self.N)
        if self.view_type == '2':
            im = im[:,:,middle_slice] #Shape (190,190)
            im =  prep_tensor(im,self.random_patch,self.num_patches,self.M,self.N)

        if self.view_type != 'multi':
            return torch.tensor(im), int(label) # output image shape [C,W,H]

        if self.view_type == 'multi':
            im0 = im[middle_slice,:,:]
            im0 =  prep_tensor(im0,self.random_patch,self.num_patches,self.M,self.N)

            im1 = im[:,middle_slice,:]
            im1 =  prep_tensor(im1,self.random_patch,self.num_patches,self.M,self.N)

            im2 =  im[:,:,middle_slice]
            im2 =  prep_tensor(im2,self.random_patch,self.num_patches,self.M,self.N)

            return torch.tensor(im0), torch.tensor(im1), torch.tensor(im2), int(label)

