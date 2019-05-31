# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class EDVR_Configs(ConfigsBase):
    def __init__(self):
        super(EDVR_Configs, self).__init__()
        nframes = 7
        self.model_configs = {
            'model_name': 'EDVR',
            'nf': 128,
            'nframes': nframes,
            'groups': 8,
            'front_RBs': 5,
            'back_RBs': 40,
            'center': None,
            'predeblur': False,
            'HR_in': False,
            'add_padding': (2, 0),
        }

        volume_k = int(nframes) / 2
        neighbor_index = list(range(-1 * volume_k, 0)) + list(range(1, volume_k + 1))
        self.dataset_configs['train']['neigbor_index'] = neighbor_index
        self.dataset_configs['val']['neigbor_index'] = neighbor_index
        self.dataset_configs['test']['neigbor_index'] = neighbor_index
