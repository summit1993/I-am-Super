# -*- coding: UTF-8 -*-
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

class FSTRN_Configs(ConfigsBase):
    def __init__(self):
        super(FSTRN_Configs, self).__init__()
        self.model_configs = {
            'model_name': 'FSTRN',
            'frb_num': 4,
            'type': 'origin',
            'has_LR_Map': True,
        }
       