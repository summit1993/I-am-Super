# -*- coding: UTF-8 -*-
from SISR import common

import torch.nn as nn


def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
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
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 
