# -*- coding: UTF-8 -*-
import os
import torch.nn as nn
from RBPN_baseNet import *

class RBPN_Model(nn.Module):
    def __init__(self, param):
        super(RBPN_Model, self).__init__()
        # Initial Feature Extraction
        self.feat0 = ConvBlock(3, param['C_l'], 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(6, param['C_m'], 3, 1, 1, activation='prelu', norm=None)
