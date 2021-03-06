# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:28:49 2019

@author: Thomas
"""
import os
os.chdir('C:\\Users\\Thomas\\Documents\\Uni_masters\\PP2')
root = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\ProteinPrediction2\\'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from CNN import SimpleCNN
import sklearn.metrics as metrics
import torch
from Plots import create_plts
params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 0}
torch.manual_seed(1)
splits = 5 # specify the number of wanted data splits, counting starts at 0,      
           # e.g 4 splits = [0,1,2,3,4] splits (5)
form = '64' # either '64'or '1024': select 64 or 1024 embbedings 
printafterepoch = 5
no_crf = True # crf or no crf
onehot = True # with or without onehotencoding
num_epochs = 51
learning_rate = 1e-3
num_classes = 3 # number of classes (currently 3: NES,NLS and no signal)
inp, outp = int(form), num_classes # size of input and output layers (+22 if onehotencoding!!)
if onehot: inp += 22
dev = torch.device('cuda') # change to 'cpu' to use cpu
class_weights = torch.FloatTensor([1/403348, 1/7093, 1/939]).to(dev)
# =============================================================================
# Main functions
# =============================================================================
def cross_valid():
    dic = opendata(form)
    labels, predicted = [],[]
    for split in range(splits):
        if split == splits:
            label_predicted_batch, out = main(split,0,form,dic)
        else:
            label_predicted_batch, out = main(split,form,dic)
        labels.append(label_predicted_batch[0])
        predicted.append(label_predicted_batch[1])
    create_plts(1, root, learning_rate, num_epochs, benchmark_crossvalid = True, 
                labels = labels, predictions = predicted, typ = str(inp))
    return labels,predicted

def main(split, form, dic):
    train_loader, val_loader = splitdata(dic, split)
    model = SimpleCNN(num_classes, inp, outp, dev).to(dev)
    _,out,_ = train(model, train_loader, val_loader, num_epochs, learning_rate, dev, class_weights)
    _, _, _, _, _, label_predicted_batch = validate(val_loader, model, dev, class_weights)
    return label_predicted_batch, out

def pltseab(dic):
    plt.rcParams.update({'font.size': 16})
    plt.figure()
    NLS = [len(dic['Sequence'][key]) for key in range(len(dic['Data']))if 1 in dic['Labels'][key]]
    NES = [len(dic['Sequence'][key]) for key in range(len(dic['Data']))if 2 in dic['Labels'][key]]
    alle = [len(dic['Sequence'][key]) for key in range(len(dic['Data']))]
    sns.kdeplot(alle, label="All proteins") 
    sns.kdeplot(NES,label="Proteins with NES")
    sns.kdeplot(NLS,label="Proteins with NLS")
    plt.xlabel('Protein length')
    plt.ylabel('Frequency')
    plt.title('Length distribution of proteins with NLS and NES')
    
# =============================================================================
# Load / preprocess data
# =============================================================================
    
def createdata(form,elmo):
    train_fasta = open(root+"nes_nls.fasta", 'r') 
    train_fasta = train_fasta.read().split('\n')
    train_data = open(root+"nes_nls.tab", 'r') 
    train_data = train_data.read().split('\n')
    train_data = [x.split('\t') for x in train_data]
    dic = {}
    tuples = [[],[],[],[]]
    i = 0
    onehotdic = onehotEncoder()
    for x in range(int((len(train_fasta)-1)/2)):
        header = train_fasta[i][1:]
        tuples[0].append(header) 
        seq = train_fasta[i+1]
        tuples[1].append(seq)
        tuples[2].append([0]*len(seq)) 
        for y in range(len(train_data)-1): #-1 because last row is empty 
            if header == train_data[y][0]:
                start = int(train_data[y][1])-1 #-1 because we start counting at 1 not 0
                end = int(train_data[y][2])
                if train_data[y][3] == 'NLS': tuples[2][x][start:end] = [1] * (end-start)
                if train_data[y][3] == 'NES': tuples[2][x][start:end] = [2] * (end-start)
        # onehot encoding of the aminoacids
        embeddings = elmo[header]
        if len(embeddings) != len(seq): 
                print('Error: Embeddings vector of ' +header+ ' is not the same size as its sequence!' )
        if onehot:
            embeddings = np.append(embeddings, np.array([onehotdic[aa] for aa in seq]), axis= 1)
        tuples[3].append(embeddings)    
        i += 2   
    dic['Header'] = tuples[0]
    dic['Sequence'] = tuples[1]
    dic['Labels'] = tuples[2]
    dic['Data'] = tuples[3]
    if onehot: pickle.dump(dic, open( root+'pickled_files\\dict_onehot'+form+'.pickle', 'wb' ))
    else: pickle.dump(dic, open( root+'pickled_files\\dict'+form+'.pickle', 'wb' ))
    return dic

def onehotEncoder(arr = ['A','R','N','D','B','C','E','Q','Z','G','H','I','L',
                         'K','M','S','F','P','T','W','Y','V','X']):
    dic = {}
    for x in range(len(arr)):
        liste = [0] * (len(arr)-1)
        if arr[x]!='X':
            liste[x] = 1
        dic[arr[x]] = liste
    return dic

def opendata(form):
    if form == '64':
        elmo = np.load(root+'\\npz\\nes_nls_64dims.npz',  mmap_mode='r' )
    elif form == '1024':
        elmo = np.load(root+'\\npz\\nes_nls.npz',  mmap_mode='r' )
    try:       
        print('Loading pickled files...')    
        if onehot: dic = pickle.load(open(root+'pickled_files\\dict_onehot'+form+'.pickle', 'rb'))               
        else: dic = pickle.load(open(root+'pickled_files\\dict'+form+'.pickle', 'rb'))   
    except (OSError, IOError):   
        print('No data found. Creating new dataframe.')        
        dic = createdata(form,elmo)
        print('Saved and Done!')
    return dic

def splitdata (dic, split):
    end = len(dic['Header'])
    ovload = (end%(splits))
    border = int((end-ovload)/(splits))
    if splits > (split+1):
        validation = {k:dic[k][border*(split):border*(split+1)]
        for k in ('Header', 'Sequence', 'Labels', 'Data')}
        train = {k:dic[k][:border*split] + dic[k][border*(split+1):end]
        for k in ('Header', 'Sequence', 'Labels', 'Data')}
    else:        
        validation = {k:dic[k][border*split:end]
        for k in ('Header', 'Sequence', 'Labels', 'Data')}
        train = {k:dic[k][border:border*split]
        for k in ('Header', 'Sequence', 'Labels', 'Data')}
    train_dataset = CustomDataset(train)
    train_loader = DataLoader(train_dataset, **params, collate_fn=my_collate)    
    val_dataset = CustomDataset(validation)
    val_loader = DataLoader(val_dataset, **params, collate_fn=my_collate) 
    return train_loader, val_loader
 
def my_collate(batch):
    seq = [item[3] for item in batch]
    longestprot = len(max(seq ,key = len))
    data = [torch.cat((item[0],torch.tensor(np.zeros([inp,longestprot-len(item[3])])).float()), 1) for item in batch]
    target = [torch.cat((item[1],torch.tensor(-np.ones(longestprot-len(item[3]))).long())) for item in batch]
    protein = [item[2] for item in batch]        
    mask = [torch.BoolTensor(np.concatenate([np.ones(len(item[3])), np.zeros(longestprot-len(item[3]))])) for item in batch]
    return [torch.stack(data),torch.stack(target),protein,seq,torch.stack(mask).bool()]
# =============================================================================
# Train / validate data
# =============================================================================

def train(model, train_loader, val_loader, num_epochs, learning_rate, dev, class_weights):
    print('Starting to learn...')
    total_step = len(train_loader)
    out_params = []
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index = -1, reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True ,weight_decay=1e-6)
    for epoch in range(num_epochs):
        loss_train_list = []  
        label_predicted_batch = [[],[],[],[]]      
        for i, (train, labels, protein, seq, mask) in enumerate(train_loader):
            # Run the forward pass      
            train, labels, mask  = train.to(dev), labels.to(dev), mask.to(dev)  
            outputs = model(train)
            if no_crf:
                loss = criterion(outputs, labels)
            else: 
                loss = criterion(outputs, labels) 
                outputs = outputs.permute(0,2,1)
                loss = -model.crf(outputs, labels, mask = mask)+loss
                
            # Backprop and perform Adam optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track the accuracy, mcc and cm                    
            if (epoch%printafterepoch) == 0: 
                if no_crf:
                    _, predicted = torch.max(outputs.data, 1)                
                    predicted = predicted.squeeze_()
                else:
                    predicted = torch.Tensor(model.crf.decode(outputs)).to(dev) 
                label_predicted_batch = orgaBatch(labels, predicted, label_predicted_batch, mask, protein, seq)
                loss_train_list.append(loss.item())                  
        # and print the results
        if (epoch%printafterepoch) == 0:
            mcc_train, cm_train, acc, rep = calcMCCbatch(label_predicted_batch[0], label_predicted_batch[1], printreport=False)
            loss_ave = sum(loss_train_list)/len(loss_train_list)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, MCC: {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, loss_ave,
                          acc, mcc_train))
            acc_valid, mcc_valid, loss_valid, cm_valid,rep_val, label_predicted_batch_val = validate(val_loader, model, dev, class_weights)
            out_params.append([mcc_train, mcc_valid, acc, acc_valid,loss_ave, loss_valid, rep, rep_val,
                               cm_train, cm_valid])
        model.train()
    return model, out_params, label_predicted_batch

def validate(val_loader, model, dev, class_weights):
    with torch.no_grad():   
        model.eval()
        label_predicted_batch = [[],[],[],[]]
        loss_list = []
        criterion = torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index = -1, reduction = 'mean')
        for i,(validation, labels, protein, seq, mask) in enumerate(val_loader):
            # preprocess outputs to correct format (1024*70*1)
            validation, labels, mask = validation.to(dev), labels.to(dev), mask.to(dev)
            outputs = model(validation)
            if no_crf:
                # use CrossEntropyloss minimalization
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
            else: 
                # apply conditional random field and decode via Vertibri algorithm
                loss = criterion(outputs, labels)
                outputs = outputs.permute(0,2,1)
                loss = -model.crf(outputs, labels, mask = mask)+loss
                predicted = torch.Tensor(model.crf.decode(outputs)).to(dev)
            # calculate quality measurements
            label_predicted_batch = orgaBatch(labels, predicted, label_predicted_batch, mask, protein, seq)
            loss_list.append(loss.item())         
        mcc, cm, acc, rep = calcMCCbatch(label_predicted_batch[0], label_predicted_batch[1])
        loss_ave = sum(loss_list)/len(loss_list)
        print('Accuracy of the model on the validation proteins is: {:.2f}%, Loss:{:.3f} and MCC is: {:.2f}'.format(acc,loss_ave,mcc))
    return acc, mcc, loss_ave, cm, rep ,label_predicted_batch

# =============================================================================
# Helper functions
# =============================================================================
    
def orgaBatch (labels, predicted, label_predicted_batch, mask, protein, seq):
    #to do: apply in validate
    labels, predicted, mask = labels.to('cpu'), predicted.to('cpu'), mask.to('cpu')
    for x in range(len(labels)):
        label_predicted_batch[0].append(list(labels[x][mask[x]].numpy()))
        if len(predicted.size()) == 1: predicted = predicted.unsqueeze(0)
        label_predicted_batch[1].append(list(predicted[x][mask[x]].numpy()))
        label_predicted_batch[2].append(protein[x])
        label_predicted_batch[3].append(seq[x])
    return label_predicted_batch

def calcMCCbatch (labels_batch, predicted_batch, printreport=True):
#   calculate MCC over given batches of an epoch in training/validation 
    x = sum(predicted_batch, [])
    y = sum(labels_batch,[])
    mcc = metrics.matthews_corrcoef(x, y)
    cm = metrics.confusion_matrix(x, y,  [0, 1, 2]) #[0, 1, 2, 3, 4, 5]) 
    acc = metrics.accuracy_score(x,y)
    report = metrics.classification_report(x,y)
    if  printreport:
        print(report)
    return mcc,cm,acc,report

# =============================================================================
# Execution
# =============================================================================
    
if __name__ == "__main__":
    #out_params, label_predicted_batch = main(validation_split, benchmark_split, form)
    labels,predicted = cross_valid()
   
