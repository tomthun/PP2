# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:28:49 2019

@author: Thomas
"""
import os
root = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\ProteinPrediction2\\'
os.chdir(root)
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({'font.size': 16})
elmo = np.load(root+'\\npz\\nes_nls_64dims.npz',  mmap_mode='r' )

def createdata(form):
    train_fasta = open(root+"nes_nls.fasta", 'r') 
    train_fasta = train_fasta.read().split('\n')
    train_data = open(root+"nes_nls.tab", 'r') 
    train_data = train_data.read().split('\n')
    train_data = [x.split('\t') for x in train_data]
    
    dic = {}
    tuples = [[],[],[],[]]
    i = 0
    for x in range(int((len(train_fasta)-1)/2)):
        tuples[0].append(train_fasta[i][1:7]) 
        tuples[1].append(train_fasta[i+1])
        tuples[2].append([0]*len(train_fasta[i+1]))
        for y in range(len(train_data)-1):
            if train_fasta[i][1:7] == train_data[y][0]:
                start = int(train_data[y][1])
                end = int(train_data[y][2])
                if train_data[y][3] == 'NLS': tuples[2][x][start:end] = [1] * (end-start)
                if train_data[y][3] == 'NES': tuples[2][x][start:end] = [2] * (end-start)
        if train_fasta[i][1:7] == 'A0A0G2':
            tuples[3].append(elmo['A0A0G2JZ79']) 
        else:
            tuples[3].append(elmo[train_fasta[i][1:7]])
        i += 2   
    dic['Header'] = tuples[0]
    dic['Sequence'] = tuples[1]
    dic['Labels'] = tuples[2]
    dic['Data'] = tuples[3]
    pickle.dump(dic, open( root+'pickled_files\\dict'+format+'.pickle', 'wb' ))
    return dic

def opendata(form):
    try:       
        print('Loading pickled benchmark files...')    
        dic = pickle.load(open(root+'pickled_files\\dict'+form+'.pickle', 'rb'))   
    except (OSError, IOError):   
        print('No data found. Creating new dataframe.')
        dic = createdata(form)
        print('Saved and Done!')
    return dic

dic = opendata('64embeddings')