# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class EDVR_Configs(ConfigsBase):
    def __init__(self, volume_k=2):
        super(EDVR_Configs, self).__init__()
        self.model_configs = {
            'model_name': 'EDVR',
            'nf': 128,
            'nframes': volume_k * 2 + 1,
            'groups': 8,
            'front_RBs': 5,
            'back_RBs': 40,
            'center': None,
            'predeblur': False,
            'HR_in': False,
            'add_padding': (2, 0),
        }

        neighbor_index = list(range(-1 * volume_k, 0)) + list(range(1, volume_k + 1))
        self.dataset_configs['train']['neigbor_index'] = neighbor_index
        self.dataset_configs['val']['neigbor_index'] = neighbor_index
        self.dataset_configs['test']['neigbor_index'] = neighbor_index
