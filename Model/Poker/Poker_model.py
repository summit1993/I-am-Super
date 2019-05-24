# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))

from Model.SISR.EDSR_model import EDSR_Model
from Model.FSTRN.FSTRN_model import FSTRN_Model

class Poker_Model(nn.Module):
    def __init__(self, args):
        super(Poker_Model, self).__init__()
        self.EDSR = EDSR_Model(args)
        self.FSTRN = FSTRN_Model(args)
    
    def forward(self, x):
        # x[0]: LR Volume; x[1]: LR_image
        f1 = self.FSTRN([x[0]])
        f2 = self.EDSR([x[1]])
        f = f1 + f2
        return f