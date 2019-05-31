# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class EDVR_Configs(ConfigsBase):
    def __init__(self):
        super(EDVR_Configs, self).__init__()
        self.model_configs = {
            'model_name': 'EDVR',
            'nf': 128,
            'nframes': 7,
            'groups': 8,
            'front_RBs': 5,
            'back_RBs': 40,
            'center': None,
            'predeblur': False,
            'HR_in': False,
        }