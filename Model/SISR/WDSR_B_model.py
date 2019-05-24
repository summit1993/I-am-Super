# -*- coding: UTF-8 -*-
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class WDSR_B_Model(nn.Module):
    def __init__(self, args):
        super(WDSR_B_Model, self).__init__()
        # hyper-params
        self.args = args
        scale = args['scale']
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(3, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=args['res_scale'], wn=wn))

        # define tail module
        tail = []
        out_feats = scale*scale*3
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(3, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = x[0]
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x
