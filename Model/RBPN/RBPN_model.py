# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
from RBPN_baseNet import ConvBlock, DeconvBlock, ResnetBlock, DBPN_Model

class RBPN_Model(nn.Module):
    def __init__(self, param):
        super(RBPN_Model, self).__init__()

        scale_factor = param['scale_factor']
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(3, param['C_l'], 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(6, param['C_m'], 3, 1, 1, activation='prelu', norm=None)
        
        # Net_sisr (DBPN)
        self.Net_sisr = DBPN_Model(param['C_l'], param['C_h'], param['DBPN_num_stages'], scale_factor)

        # Net_misr (ResNet1)
        modules_body1 = [ResnetBlock(param['C_m'], kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) 
                            for _ in range(param['n_resblock'])]
        modules_body1.append(DeconvBlock(param['C_m'], param['C_h'], kernel, stride, padding, activation='prelu', norm=None))
        self.Net_misr = nn.Sequential(*modules_body1)

        # Net_res (ResNet2)
        modules_body2 = [ResnetBlock(param['C_h'], kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) 
                            for _ in range(param['n_resblock'])]
        modules_body2.append(ConvBlock(param['C_h'], param['C_h'], 3, 1, 1, activation='prelu', norm=None))
        self.Net_res = nn.Sequential(*modules_body2)

        # Net_D (ResNet3)
        modules_body3 = [ResnetBlock(param['C_h'], kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
                            for _ in range(param['n_resblock'])]
        modules_body3.append(ConvBlock(param['C_h'], param['C_l'], kernel, stride, padding, activation='prelu', norm=None))
        self.Net_D = nn.Sequential(*modules_body3)

        #Reconstruction
        self.output = ConvBlock((param['nFrames']-1)*param['C_h'], 3, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_() 
    
    def forward(self, inputs):
        # inputs[0]: LR image; inputs[1]: neighbors
        x = inputs[0]
        neigbor = inputs[1]
        neigbor = neigbor.permute(1, 0, 2, 3)
        # Initial Feature Extraction
        feat_input = self.feat0(x)
        feat_frame=[]
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j]),1)))
        
        # Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.Net_sisr(feat_input)
            h1 = self.Net_misr(feat_frame[j])
            
            e = h0-h1
            e = self.Net_res(e)
            h = h0+e
            Ht.append(h)
            feat_input = self.Net_D(h)
        
        # Reconstruction
        out = torch.cat(Ht,1)        
        output = self.output(out)

        return output
