#This script takes the log file for the loss weight sweep for the given dataset and plots the performance plot across weights

import os
import matplotlib.pyplot as plt 
import pandas as pd

log_path = '/home/skh259/LinLab/LinLab/CAT-XPLAIN/logs'

dataset = "cifar"
log_file = os.path.join(log_path,dataset+'_two_classes_expViT_loss_weight_sweep.log')
with open(log_file,'r') as infile:
    text = infile.read()

text = text.strip().split('\n')

weights_list = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90,1.0]
fracs = [0.05]*8+[0.10]*8+[0.25]*8+[0.50]*8+[0.75]*8
loss_weights = weights_list+weights_list+weights_list+weights_list+weights_list

df = pd.DataFrame(columns=['fracs','loss_weights','ph_acc','ace','avg','test_ph_acc','test_ace','test_full_acc'])
df['fracs'] = fracs
df['loss_weights'] = loss_weights

ph_acc = []
ace = []
avg = []
test_ph_acc = []
test_ace = []
test_full_acc = []

for line in text:
    if line.startswith("mean val ph acc: "):
        p = float(line.split("mean val ph acc: ")[1].strip().split(' ')[0])
        ph_acc.append(p)
    if line.startswith("mean val ICE: "):
        a = float(line.split("mean val ICE: ")[1].strip().split(' ')[0])
        ace.append(a)

    if line.startswith("mean test ph acc: "):
        test_ph_acc.append(float(line.split("mean test ph acc: ")[1].strip().split(' ')[0]))
    if line.startswith("mean test ICE: "):
        test_ace.append(float(line.split("mean test ICE: ")[1].strip().split(' ')[0]))
    if line.startswith("mean test true acc with whole input: "):
        test_full_acc.append(float(line.split("mean test true acc with whole input: ")[1].strip().split(' ')[0]))

df['ph_acc'] = ph_acc
df['ace'] = ace
df['avg'] = [(ph_acc[i]+ace[i])/2 for i in range(len(ph_acc))]
df['test_ph_acc'] = test_ph_acc
df['test_ace'] = test_ace
df['test_full_acc'] = test_full_acc

save_path = os.path.join(log_path,dataset+'_expViT_loss_sweep.csv')
df.to_csv(save_path)

def results_plot(log_path,dataset,frac,ph_acc,ace): 
    # line 1 points
    x1 = weights_list
    y1 = ph_acc
    # plotting the line 1 points 
    plt.plot(x1, y1, label = "val_ph_acc")
    # line 2 points
    x2 = weights_list
    y2 = ace
    # plotting the line 2 points 
    plt.plot(x2, y2, label = "val_ace")
    plt.xlabel('loss_weight')
    # Set the y axis label of the current axis.
    plt.ylabel('performance')
    # Set a title of the current axes.
    plt.title('Loss weight sweep frac_patches:'+str(frac)+" "+dataset)
    # show a legend on the plot
    plt.legend()
    # save a figure.
    plt.savefig(os.path.join(log_path,dataset+'_frac_'+str(frac)+'_loss_sweep.png'), bbox_inches='tight')
    plt.close()


FRACS = [0.05,0.10,0.25,0.50,0.75]
for f in FRACS:
    frac_result = df[df['fracs']==f]

    results_plot(log_path,dataset,f,list(frac_result['ph_acc']),list(frac_result['ace']))
    






