# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(sys.path[0], '../..'))
from Model.FSTRN import FSTRN_baseNet

class FSTRN_Model(nn.Module):
    def __init__(self, model_para):
        super(FSTRN_Model, self).__init__()
        self.model_para = model_para
        # self.lfe = self.create_bottle_net(has_relu=False)
        self.lfe = FSTRN_baseNet.FRB_Block(has_relu=False, type=model_para['type'])
        self.FRB_blocks = nn.Sequential()
        self.frb_num = model_para['frb_num']
        for i in range(self.frb_num):
            # self.FRB_blocks.add_module('rfb_' + str(i + 1), self.create_bottle_net())
            self.FRB_blocks.add_module('rfb_' + str(i + 1), FSTRN_baseNet.FRB_Block(has_relu=True, 
                type=model_para['type']))
        self.lrl = nn.Sequential()
        self.lrl.add_module('PReLU', nn.PReLU())
        self.lsr = nn.Sequential()
        self.lsr.add_module('conv1', nn.Conv2d(3, 3, 3, 1, 1))
        # H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
        self.lsr.add_module('dconv', nn.ConvTranspose2d(3, 3, kernel_size=8, stride=4, padding=2, output_padding=0))
        self.lsr.add_module('conv2', nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        # x[0]: LR Volume; x[1]: up-sampled LR image
        f_0 = self.lfe(x[0])
        f = f_0 + self.FRB_blocks.__getattr__('rfb_1')(f_0)
        for i in range(1, self.frb_num):
            f = f + self.FRB_blocks.__getattr__('rfb_' + str(i + 1))(f)
        f += f_0
        f = torch.sum(f, 2)
        f = self.lrl(f)
        f = self.lsr(f)
        if self.model_para['has_LR_Map']:
            f += x[1]
        return f