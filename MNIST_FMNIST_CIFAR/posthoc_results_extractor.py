#This script extracts results for all fractions 

import os
import pandas as pd
import numpy as np
from config import plots_path, log_path

datasets = ["mnist","fmnist","cifar"]
for dataset in datasets:
    fractions = [0.05,0.10,0.25,0.50]
    df = pd.DataFrame(columns=['fracs','ph_acc','ace'])
    log_file = os.path.join(log_path,dataset+"_two_classes_posthoc_results.log")
    with open(log_file,'r') as infile:
        text = infile.read()

    text = text.strip().split('\n')
    ph_acc = []
    ace = []
    for t in text:
        if "mean test ph acc: " in t:
            ph_acc.append(t.strip())
        if "mean test ICE: " in t:
            ace.append(t.strip())

    df['ph_acc'] = ph_acc
    df['ace'] = ace
    df['fracs'] = fractions
    print(df)
    df_file = os.path.join(plots_path,dataset+"_two_classes_posthoc_results.csv")
    df.to_csv(df_file)