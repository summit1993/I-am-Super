# -*- coding: UTF-8 -*-
import sys, os
sys.path.append(os.path.join(sys.path[0], '../..'))
from Model.SISR import common
import torch.nn as nn

class EDSR_Model(nn.Module):
    def __init__(self, args):
        super(EDSR_Model, self).__init__()
        conv = common.default_conv
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3 
        # upsample factor
        scale = args['scale']
        act = nn.ReLU(True)
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args['res_scale']
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = x[0]
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 
