# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch

class FRB_Block(nn.Module):
    def __init__(self, has_relu=True, type='origin'):
        self.has_relu = has_relu
        self.type = type
        super(FRB_Block, self).__init__()
        if self.has_relu:
            self.prelu = nn.PReLU()
        if self.type == 'origin':
            self.SptioConv = nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1))
            self.TemporalConv =  nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0))
        else:
            self.SptioConv_3 = nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1))
            self.SptioConv_5 = nn.Conv3d(3, 3, (1, 5, 5), 1, (0, 2, 2))
            self.TemporalConv_3 =  nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0))
            self.TemporalConv_5 =  nn.Conv3d(3, 3, (5, 1, 1), 1, (2, 0, 0))
            self.concatConv = nn.Conv3d(20, 5, (1,1,1), 1, (0, 0, 0))
    def forward(self, x):
        if self.has_relu:
            x = self.prelu(x)
        if self.type == 'origin':
            x = self.SptioConv(x)
            x = self.TemporalConv(x)
        else:
            x_3 = self.SptioConv_3(x)
            x_5 = self.SptioConv_5(x)
            x_3_3 = self.TemporalConv_3(x_3)
            x_3_5 = self.TemporalConv_5(x_3)
            x_5_3 = self.TemporalConv_3(x_5)
            x_5_5 = self.TemporalConv_5(x_5)
            x_cat = torch.cat([x_3_3, x_3_5, x_5_3, x_5_5], 2)
            x_cat = x_cat.permute(0, 2, 1, 3, 4)
            x = self.concatConv(x_cat)
            x = x.permute(0, 2, 1, 3, 4)
        return x

def create_bottle_net(self, has_relu=True, type='origin'):
        bottle_net = nn.Sequential()
        if has_relu:
            bottle_net.add_module('PReLU', nn.PReLU())
        if type == 'origin':
            bottle_net.add_module('SptioConv', nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1)))
            bottle_net.add_module('TemporalConv', nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0)))
        else:
            bottle_net.add_module('SptioConv_3', nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1)))
            bottle_net.add_module('SptioConv_5', nn.Conv5d())
            bottle_net.add_module('TemporalConv', nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0)))
        return bottle_net