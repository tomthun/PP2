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
                nn.Conv2d(inp, 128, kernel_size=(1,1),  padding = (0,0)),
                nn.LeakyReLU(), 
                nn.Dropout(0.3)
                #nn.BatchNorm2d(128)
                ) 
#        self.attentionlayer = nn.TransformerEncoder(
#                nn.TransformerEncoderLayer(d_model=128, nhead=1),
#                num_layers=1
#                )
        #self.rnn = nn.LSTM(128,128, num_layers = 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(9,1),  padding = (4,0)), 
            nn.LeakyReLU(), 
            nn.Dropout(0.3)
            #nn.BatchNorm2d(64)
            ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, outp, kernel_size=(5,1),  padding=(2,0))
            )     
        #self.crf = CRF(num_tags,batch_first=True).to(dev)
    
    def forward(self, x):
        out = self.compLayer(x.unsqueeze(3))
        #out, (hn,cn) = self.rnn(out.squeeze(3).permute(2,0,1))
#        out = self.attentionlayer(out.squeeze(dim = 3).permute(2,0,1))
        out = self.layer1(out)#.permute(1,2,0).unsqueeze(3))
        out = self.layer2(out)
        out = out.squeeze(dim = 3)
        return out
