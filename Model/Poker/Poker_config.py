# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class Poker_Configs(ConfigsBase):
    def __init__(self):
        super(Poker_Configs, self).__init__()
        self.model_configs = {
            'model_name': 'Poker',
            # FSTRN configs
            'frb_num': 4,
            'type': 'poker',
            'has_LR_Map': False,
            # EDSR configs
            'scale': 4,
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1.0,
        }