#This script takes the log file for the loss weight sweep for the given dataset and plots the performance plot across weights

import os
import matplotlib.pyplot as plt 
import pandas as pd
from config import plots_path, log_path

def results_plot(log_path,dataset,frac,ph_acc,ace,acc): 
    # line 1 points
    x1 = weights_list
    y1 = ph_acc
    # plotting the line 1 points 
    plt.plot(x1, y1, label = "ph_acc")
    # line 2 points
    x2 = weights_list
    y2 = ace
    # plotting the line 2 points 
    plt.plot(x2, y2, label = "ace")

    # line 3 points
    x3 = weights_list
    y3 = acc
    # plotting the line 3 points 
    plt.plot(x3, y3, label = "acc")

    plt.xlabel('$\lambda$')

    # Set the y axis label of the current axis.
    plt.ylabel('Validation performance')
    # Set a title of the current axes.
    plt.title('Loss weight sweep frac_patches:'+str(frac)+" "+dataset)
    # show a legend on the plot
    plt.legend(loc = "upper left")
    # save a figure.
    if all_metrics:
        plt.savefig(os.path.join(plots_path,dataset+'_frac_'+str(frac)+'_loss_sweep_full_metrics.png'), bbox_inches='tight',dpi=1000)
    else:
        plt.savefig(os.path.join(plots_path,dataset+'_frac_'+str(frac)+'_loss_sweep.png'), bbox_inches='tight',dpi=1000)
    plt.close()

datasets = ["mnist","fmnist","cifar"]
for dataset in datasets:
    all_metrics = True
    if all_metrics:
        log_file = os.path.join(log_path,"all_metrics_"+dataset+'_two_classes_expViT_loss_weight_sweep.log')
    else:
        log_file = os.path.join(log_path,dataset+'_two_classes_expViT_loss_weight_sweep.log')
    with open(log_file,'r') as infile:
        text = infile.read()

    text = text.strip().split('\n')

    weights_list = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90,1.0]
    fracs = ["0.05"]*8+["0.10"]*8+["0.25"]*8+["0.50"]*8
    loss_weights = weights_list+weights_list+weights_list+weights_list

    df = pd.DataFrame(columns=['fracs','loss_weights','ph_acc','ace','acc','avg','test_ph_acc','test_ace','test_full_acc'])
    df['fracs'] = fracs
    df['loss_weights'] = loss_weights
    ph_acc = []
    ace = []
    acc = []
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
        if all_metrics:
            if line.startswith("mean val true acc with whole input: "):
                a = float(line.split("mean val true acc with whole input: ")[1].strip().split(' ')[0])
                acc.append(a)

        if line.startswith("mean test ph acc: "):
            test_ph_acc.append(float(line.split("mean test ph acc: ")[1].strip().split(' ')[0]))
        if line.startswith("mean test ICE: "):
            test_ace.append(float(line.split("mean test ICE: ")[1].strip().split(' ')[0]))
        if line.startswith("mean test true acc with whole input: "):
            test_full_acc.append(float(line.split("mean test true acc with whole input: ")[1].strip().split(' ')[0]))

    df['ph_acc'] = ph_acc
    df['ace'] = ace
    df['acc'] = acc
    if all_metrics:
        df['avg'] = [(ph_acc[i]+ace[i]+acc[i])/3 for i in range(len(ph_acc))]
    else:   
        df['avg'] = [(ph_acc[i]+ace[i])/2 for i in range(len(ph_acc))]
    df['test_ph_acc'] = test_ph_acc
    df['test_ace'] = test_ace
    df['test_full_acc'] = test_full_acc

    if all_metrics:
        save_path = os.path.join(plots_path,dataset+'_expViT_loss_sweep_full_metrics.csv')
    else:
        save_path = os.path.join(plots_path,dataset+'_expViT_loss_sweep.csv')
    df.to_csv(save_path)

    FRACS = ["0.05","0.10","0.25","0.50"]
    df_results = pd.DataFrame(columns=['fracs','loss_weight','test_ph_acc','test_ace','test_full_acc'])
    df_results['fracs'] = FRACS
    val_results = pd.DataFrame(columns=['fracs','loss_weight','val_ph_acc','val_ace','val_full_acc'])
    val_results['fracs'] = FRACS
    loss_weight = []
    t_ph_acc = []
    t_ace = []
    t_acc = []
    v_ph_acc = []
    v_ace = []
    v_acc = []
    for f in FRACS:
        max_avg = df[df['fracs']==f]['avg'].max()
        best_performance = df[(df['fracs']==f) & (df['avg']==max_avg)]
        loss_weight.append(best_performance['loss_weights'].item())
        t_ph_acc.append(best_performance['test_ph_acc'].item())
        t_ace.append(best_performance['test_ace'].item())
        t_acc.append(best_performance['test_full_acc'].item())
        v_ph_acc.append(best_performance['ph_acc'].item())
        v_ace.append(best_performance['ace'].item())
        v_acc.append(best_performance['acc'].item())

    
    df_results['fracs'] = FRACS
    df_results['loss_weight'] = loss_weight
    df_results['test_ph_acc'] = t_ph_acc
    df_results['test_ace'] = t_ace
    df_results['test_full_acc'] = t_acc
    df_results['ph_acc'] = v_ph_acc
    df_results['ace'] = v_ace
    df_results['acc'] = v_acc


    if all_metrics:
        save_path = os.path.join(plots_path,dataset+'_expViT_results_full_metrics.csv')
    else:
        save_path = os.path.join(plots_path,dataset+'_expViT_results.csv')
    df_results_test = df_results[['fracs','loss_weight','test_ph_acc','test_ace','test_full_acc']]
    df_results_test.to_csv(save_path)


    for f in FRACS:
        frac_result = df[df['fracs']==f]
        ph_acc = list(frac_result['ph_acc'])
        ace = list(frac_result['ace'])
        acc = list(frac_result['acc'])

        results_plot(plots_path,dataset,f,ph_acc,ace,acc)
        






