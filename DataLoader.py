# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:47:10 2019

@author: Thomas
"""

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dic):
        self.dic = dic
     
    def __len__(self):
        return len(self.dic['Header'])   
    
    def __getitem__(self,idx):
        data = torch.tensor(self.dic['Data'][idx]).float().permute(1,0)
        labels = torch.tensor(self.dic['Labels'][idx]).long()
        protein = self.dic['Header'][idx]
        seq = self.dic['Sequence'][idx]
        return data,labels,protein,seq