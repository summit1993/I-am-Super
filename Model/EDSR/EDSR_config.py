# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class EDSR_Configs(ConfigsBase):
    def __init__(self):
        super(EDSR_Configs, self).__init__()
        self.model_configs = {
            'model_name': 'EDSR',
            'scale': 4,
            'n_resblocks': 16,
            'n_feats': 64,
            'res_scale': 1.0,
        }
        neighbor_index = []
        self.dataset_configs['train']['neigbor_index'] = neighbor_index
        self.dataset_configs['val']['neigbor_index'] = neighbor_index
        self.dataset_configs['test']['neigbor_index'] = neighbor_index