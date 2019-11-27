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

        self.layer1 = nn.Sequential(
            nn.Conv2d(inp, 64, kernel_size=(7,1),  padding = (3,0)), nn.LeakyReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(64)) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, outp, kernel_size=(3,1),  padding=(1,0)), nn.LeakyReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(outp)) 
        self.crf = CRF(num_tags,batch_first=True).to(dev)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
