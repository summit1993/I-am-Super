# -*- coding: UTF-8 -*-
import pickle
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.config import ConfigsBase

# root_dir = '/data1/youku'
# root_dir = 'D:\\program\\deep_learning\\I-am-Super\\data'
root_dir = 'H:\\data\\youku'

class RBPN_Configs(ConfigsBase):
    def __init__(self):
        super(RBPN_Configs, self).__init__()
        self.nFrames = 7,
        self.model_configs = {
            'model_name': 'RBPN', 
            'scale_factor': 4,
            'C_l': 256, 
            'C_m': 256,
            'C_h': 64,
            'DBPN_num_stages': 3,
            'n_resblock': 5, 
            'nFrames': self.nFrames,
        }
        train_tmp = pickle.load(open(os.path.join(root_dir, 'train.pkl'), 'rb'))
        val_tmp = pickle.load(open(os.path.join(root_dir, 'val.pkl'), 'rb'))
        volume_k = int(self.nFrames / 2)
        