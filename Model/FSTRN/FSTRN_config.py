# -*- coding: UTF-8 -*-
import pickle
import os
from Core.config import ConfigsBase

# root_dir = '/data1/youku'
root_dir = 'D:\\program\\deep_learning\\Deep-HC\\I-am-Fine\\VSR\\data'

class FSTRN_Configs(ConfigsBase):
    def __init__(self):
        super(FSTRN_Configs, self).__init__()
        self.model_configs = {
            'rfb_num': 4,
        }
        train_tmp = pickle.load(open(os.path.join(root_dir, 'train.pkl'), 'rb'))
        val_tmp = pickle.load(open(os.path.join(root_dir, 'val.pkl'), 'rb'))
        volume_k = 2
        self.dataset_configs = {
            'train': {'shuffle': True, 'transform': self._get_transform(),
                      'images': train_tmp, 'volume_k': volume_k, 'has_hr': True,
                      'image_root_dir': os.path.join(root_dir, 'images')},
            'val': {'shuffle': False, 'transform': self._get_transform(),
                    'images': val_tmp, 'volume_k': volume_k, 'has_hr': True,
                    'image_root_dir': os.path.join(root_dir, 'images')},
        }