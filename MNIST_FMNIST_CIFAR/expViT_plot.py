#This script takes the log file for the loss weight sweep for the given dataset and plots the performance plot across weights

import os
import matplotlib.pyplot as plt 
import pandas as pd

log_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/logs'

dataset = "mnist"
log_file = os.path.join(log_path,dataset+'_two_classes_expViT_loss_weight_sweep.log')
with open(log_file,'r') as infile:
    text = text.read()

frac_list = [0.05,0.10,0.25,0.50,0.75]
weights_list = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90,1.0]

#eg:
# mean test ph acc: 0.934 , std dev: 0.031 
# mean test ICE: 0.023 , std dev: 0.016 
# mean test true acc with whole input: 0.980 , std dev: 0.016 

fracs = text.split("frac")[1:]
for i in range(len(fracs)):
    weights = fracs[i].split("DONE!!!!!!!")[:-1]
    for j in range(len(weights)):
        t = weights[j].strip().split('\n')
        for line in t:
            if line.startswith("mean test ph acc: "):
                ph_acc = float(line.split("mean test ph acc: ")[0].strip())
            if line.startswith("mean test ICE: "):
                ace = float(line.split("mean test ICE: ")[0].strip())
            if line.startswith("mean test true acc with whole input: "):
                acc = float(line.split("mean test true acc with whole input: ")[0].strip())





