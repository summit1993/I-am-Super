# -*- coding: UTF-8 -*-
import pickle
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class RBPN_Configs(ConfigsBase):
    def __init__(self):
        super(RBPN_Configs, self).__init__()
        self.nFrames = 7 # 5
        self.model_configs = {
            'model_name': 'RBPN', 
            'scale_factor': 4,
            'C_l': 32, # 256 
            'C_m': 32, # 256
            'C_h': 4,  # 64
            'DBPN_num_stages': 3, # 4
            'n_resblock': 4,  # 5
            'nFrames': self.nFrames,
        }

        volume_k = int(self.nFrames / 2)
        neighbor_index = list(range(-1 * volume_k, 0)) + list(range(1, volume_k + 1))
        self.dataset_configs['train']['neigbor_index'] = neighbor_index
        self.dataset_configs['val']['neigbor_index'] = neighbor_index
        