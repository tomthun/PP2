# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
root = "C:\\Users\\Thomas\\Documents\\Uni_masters\\PP2\\"
np.load(root + data_name)
train_fasta = open(root+"nes_nls.fasta", 'r') 
train_fasta = train_fasta.read().split('\n')
train_data = open(root+"nes_nls.tab", 'r') 
train_data = train_data.read().split('\n')
train_data = [x.split('\t') for x in train_data]