# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:38:41 2019

@author: Thomas
"""

from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import random
import sklearn.metrics as metrics
from operator import add 
from statistics import mean
from pltCM import pltConfusionMatrix
# =============================================================================
# Functions to create plots
# =============================================================================
def create_plts( split, root, learning_rate, num_epochs,out_params=[], mcc_orga = 0, cm_orga = 0,
                 benchmark_crossvalid = False, labels = [], predictions = [], typ = ''):
    split = str(split)
    c = ['Others(non-signal)', 'NES', 'NLS'] 
    if benchmark_crossvalid:
        mccs, accs, cms = randSampler(labels,predictions)
        cm_mean, cm_standard  = meanstdCM(cms)
#        cm_mean_res, cm_standard_res  = meanstdCM(cms_res)
#        boxplt (mccs, mccs_res , 'MCC', root, learning_rate,num_epochs,typ)
#        boxplt (accs, acc_res, 'accuracy', root, learning_rate,num_epochs,typ)
        #plot_bar_csrel(out_params,root,split,learning_rate,num_epochs)
        pltConfusionMatrix(cm_mean, c)
        plt.savefig(root + 'Pictures\\'+ 'CM'+ typ +'_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_plot.png')
    else:        
        #------------------------------Loss------------------------------
        loss_val, loss_train, epochs, acc_val, acc_train, mcc_val, mcc_train = (np.array([x[0] for x in out_params]),
        np.array([x[1] for x in out_params]), np.array([x[2] for x in out_params]), np.array([x[3] for x in out_params]),
        np.array([x[4] for x in out_params]), np.array([x[5] for x in out_params]), np.array([x[6] for x in out_params]))
        x = np.linspace(epochs.min(),epochs.max(),500)
        loss_val, loss_train, acc_val, acc_train, mcc_val, mcc_train = (make_interp_spline(epochs, loss_val, k=3), make_interp_spline(epochs, loss_train, k=3),
        make_interp_spline(epochs, acc_val, k=3), make_interp_spline(epochs, acc_train, k=3),
        make_interp_spline(epochs, mcc_val, k=3), make_interp_spline(epochs, mcc_train, k=3))
        plt.plot(x, loss_val(x),  label='Loss of the validation data')
        plt.plot(x, loss_train(x),  label='Loss of the train data')
        plt.legend()
        plt.title('Train vs validation loss')
        plt.xlabel('Number of epochs')
        plt.ylabel('Model loss') 
        plt.savefig(root + 'Pictures\\loss_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Accuracy------------------------------
        plt.plot(x, acc_val(x),  label='Accuracy on the validation split ' + split)
        plt.plot(x, acc_train(x),  label='Accuracy on the train data')
        plt.legend()
        plt.title('Train vs validation  accuracy' + split)
        plt.xlabel('Number of epochs')
        plt.ylabel('Model accuracy in %') 
        plt.savefig(root + 'Pictures\\acc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------MCC------------------------------
        plt.plot(x, mcc_val(x),  label='MCC of the validation split ' + split)
        plt.plot(x, mcc_train(x),  label='MCC of the train data')
        plt.legend()
        plt.title('Train vs validation MCC')
        plt.xlabel('Number of epochs')
        plt.ylabel('Model MCC')
        plt.savefig(root + 'Pictures\\mcc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Confusion matrix------------------------------
        last_entry = out_params[len(out_params)-1]
        cm_valid, cm_train = last_entry[len(last_entry)-1] , last_entry[len(last_entry)-2]
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix trainset, without normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix of validationset split '+split+', without normalization')
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix trainset, with normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix validationset split '+split+', with normalization')
    

def boxplt (glo, res, title, root, learning_rate,num_epochs,typ):
    plt.rcParams["figure.figsize"] = [6,6]
    plt.subplot(1, 2, 1)
    post, post_std = np.mean(glo), np.std(glo)
    plt.boxplot(glo)
    plt.title("Boxplot of the\nglobal "+title)
    fmt = '.3f'
    plt.xticks([1], ['Mean: '+ format(post, fmt)  +' +/- ' + format(post_std, fmt)])   
    
    plt.subplot(1, 2, 2)
    post, post_std = np.mean(res), np.std(res)
    plt.boxplot(res)
    plt.title("Boxplot of the\nresidue "+title)
    fmt = '.3f'
    plt.xticks([1], ['Mean: '+ format(post, fmt)  +' +/- ' + format(post_std, fmt)])   
    
    plt.tight_layout()
    plt.savefig(root + 'Pictures\\'+ title+typ +'_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_boxplot.png')
    plt.close()
    plt.rcParams["figure.figsize"] = [9,6]
    
def calcSTDandMEANplot(out_params, x, y, param, root, learning_rate, num_epochs):
    mean_valid = []
    std_valid = []
    mean_train = []
    std_train = []
    epochs = []
    for idx in range(len(out_params[0])):
        std_valid.append(np.std([split[idx][x] for split in out_params]))
        mean_valid.append(np.mean([split[idx][x] for split in out_params]))
        std_train.append(np.std([split[idx][y] for split in out_params]))
        mean_train.append(np.mean([split[idx][y] for split in out_params]))
        epochs.append(out_params[0][idx][2])
    mean_valid, std_valid, mean_train, std_train, epochs = (np.array(mean_valid),
    np.array(std_valid), np.array(mean_train), np.array(std_train), np.array(epochs))  
    smooth = np.linspace(epochs.min(),epochs.max(),500)
    funcx = (make_interp_spline(epochs, mean_valid, k=3))
    funcy = (make_interp_spline(epochs, mean_train, k=3))
    funcstdx = (make_interp_spline(epochs, std_valid, k=3))
    funcstdy = (make_interp_spline(epochs, std_train, k=3))
    plt.figure()
    plt.plot(smooth, funcx(smooth),  label='Standardized '+ param +' of the validation data')
    plt.fill_between(smooth, funcx(smooth)-funcstdx(smooth), funcx(smooth)+funcstdx(smooth), alpha=0.5)
    plt.plot(smooth, funcy(smooth),  label='Standardized '+ param +' of the trainings data')
    plt.fill_between(smooth, funcy(smooth)-funcstdy(smooth), funcy(smooth)+funcstdy(smooth), alpha=0.5)
    plt.legend()
    plt.title('Standardized '+ param)
    plt.xlabel('Number of epochs')
    plt.ylabel('Model '+ param) 
    plt.savefig(root + 'Pictures\\Standardized_'+ param +'_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_plot.png')
    plt.close()
    
def plot_bar_csrel(out_params,root,split,learning_rate,num_epochs):
    cs_pre, cs_pre_std = np.mean([x[4] for x in out_params]), np.std([x[4] for x in out_params])
    cs_post, cs_post_std = np.mean([x[5] for x in out_params]), np.std([x[4] for x in out_params])
    cpre = Counter(sum([x[6] for x in out_params],[]))
    cpost = Counter(sum([x[7] for x in out_params],[]))
    labels = ['x<-5','-5','-4','-3','-2','-1','0','1','2','3','4','5','x>5']
    lsrpre,lsrpost = [i for i in cpre if i < -5], [i for i in cpre if i < -5]
    gr8pre,gr8post = [i for i in cpost if i > 5], [i for i in cpost if i > 5]
    pre = np.log([len(lsrpre),cpre[-5],cpre[-4],cpre[-3],cpre[-2],cpre[-1],cpre[0],cpre[1],cpre[2],cpre[3],cpre[4],cpre[5],len(gr8pre)])
    post = np.log([len(lsrpost),cpost[-5],cpost[-4],cpost[-3],cpost[-2],cpost[-1],cpost[0],cpost[1],cpost[2],cpost[3],cpost[4],cpost[5],len(gr8post)])
    fig, ax = plt.subplots()   
    x = np.arange(len(labels))
    width = 0.35 
    ax.bar(x - width/2, pre, width, label='Non-processing')
    ax.bar(x + width/2, post, width, label='Post-processing')
    ax.set_ylabel('Log-occurances')
    ax.set_title('Relative deviation of the real cleavage site')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fmt = '.3f'
    ax.set_xlabel('Relative distance to the real cleavage site\n'+
                  'Mean deviation non-processing: '+format(cs_pre, fmt)+' +/- '+format(cs_pre_std, fmt)+'\n'+
                  'Mean deviation post-processing: '+format(cs_post, fmt)+' +/- '+format(cs_post_std, fmt))
    ax.legend()
    plt.tight_layout()
    plt.savefig(root + 'Pictures\\Standardized_cleavagesite_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_plot.png')
    plt.close()
    
def plot_confusion_matrix (cm, classes, root, learning_rate, num_epochs, split, normalize=False, title=None, 
                          cmap=plt.cm.Blues, cm_standard = 0, cm_benchmark = False):
    plt.rcParams.update({'font.size': 8})

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm_norm = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm_norm
        if cm_benchmark:
            cm_standard = cm_standard.astype('float') / cm_norm
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'f'
    thresh = cm.max() / 2.
    if cm_benchmark:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt) + " +/- " + format(cm_standard[i,j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    else: 
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
    plt.savefig(root + 'Pictures\\' + title + '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+str(split)+'.png')
    return ax

def meanstdCM(cms):
    cm_mean = 0
    cm_standard = 0
    for cm in cms:
        cm_mean += cm
    cm_mean = cm_mean/len(cms)
    for cm in cms:
        cm_standard += (cm_mean - cm)**2
    cm_standard = cm_standard/4
    cm_standard = (cm_standard**0.5)/len(cms)**0.5
    return cm_mean, cm_standard

def randSampler (l, p):
    mccs = []
    accs = []
    cms = []    
    per_resi_res = perRes(l,p)        
    for x in range(10000):
        val = []
        for y in range(len(l)):
            r_indice = random.sample(range(len(l)), 1)
            val.append(per_resi_res[r_indice [0]])
        mccs.append(sum([x[0] for x in val])/len(val))
        accs.append(sum([x[1] for x in val])/len(val))
        cms.append(sum([x[2] for x in val]))
    return mccs, accs, cms

def perRes (labels, predictions):
    res = []
    for x in range(len(labels)):
        mcc = metrics.matthews_corrcoef(labels[x], predictions[x])
        acc = metrics.accuracy_score(labels[x],predictions[x])  
        cm = metrics.confusion_matrix(labels[x], predictions[x],  [0, 1, 2])
        res.append([mcc,acc,cm])
    return res

def csdiff(labels, predictions):
    csdiff = 0
    for x in range (len(labels)):
        csdiff += abs(labels[x].count(0) - predictions[x].count(0))
    csdiff = csdiff/len(labels)
    return csdiff   
 
