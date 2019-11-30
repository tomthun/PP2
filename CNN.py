# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 23:03:08 2019

@author: Thomas
"""

import torch.nn as nn
from torchcrf import CRF
class SimpleCNN(nn.Module):
    
    # ToDo adapt CNN
    def __init__(self,num_tags,inp,outp,dev):
        super(SimpleCNN, self).__init__()
        self.compLayer = nn.Sequential(
            nn.Conv2d(inp, 512, kernel_size=(1,1),  padding = (0,0)), nn.LeakyReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(512)) 
        self.attentionlayer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=1), num_layers=3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=(7,1),  padding = (3,0)), nn.LeakyReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(64)) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, outp, kernel_size=(3,1),  padding=(1,0)), nn.LeakyReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(outp)) 
        self.crf = CRF(num_tags,batch_first=True).to(dev)
    
    def forward(self, x):
        out = self.compLayer(x.unsqueeze(3))
        out = self.attentionlayer(out.squeeze(dim = 3).permute(2,0,1))
        out = self.layer1(out.permute(1,2,0).unsqueeze(3))
        out = self.layer2(out)
        return out
