#This script extracts val_acc of base model for all hyperparameters

import os
import pandas as pd
import numpy as np
from config import plots_path, log_path

datasets = ["mnist","fmnist","cifar"]
for dataset in datasets:
    log_file = os.path.join(log_path,dataset+"_two_classes_posthoc_hypersweep.log")
    with open(log_file,'r') as infile:
        text = infile.read()

    text = text.strip().split('\n')
    depths = [2,2,2,2,4,4,4,4,6,6,6,6,8,8,8,8]
    dim = [64,128,256,512,64,128,256,512,64,128,256,512,64,128,256,512]
    acc = []
    df = pd.DataFrame(columns=['depth','dim','val_acc'])
    df['depth'] = depths
    df['dim'] = dim

    for t in text:
        if "Model Accuracy = tensor" in t:
            acc.append(float(t.split('(')[1].split(",")[0]))
    df['val_acc'] = acc
    best_parameter = np.argmax(df['val_acc'])
    print(df)
    print("for",log_file)
    print("Best depth,dim:val_acc")
    print(df.loc[[best_parameter]])
    df_file = os.path.join(plots_path,dataset+"_two_classes_posthoc_hypersweep.csv")
    df.to_csv(df_file)