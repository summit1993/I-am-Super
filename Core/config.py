# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import torch
import numpy as np
import pickle
import os

# root_dir = '/data1/youku'
# root_dir = 'D:\\program\\deep_learning\\I-am-Super\\data'
root_dir = 'H:\\data\\youku'

class ConfigsBase:
    def __init__(self):
        self.regular_configs = {
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'batch_size': 16,
            'num_workers': 10,
            'epoch_num': 20,
            'show_iters': 10,
            'model_save_epoch': 1,
        }
        self.optimizer_configs = {
            'lr': 1e-5,
            'weight_decay': 1e-5,
            'criterion': torch.nn.MSELoss(),
            'lr_decay_step': 1,
            'lr_decay_factor': 0.1,
        }
        self.model_configs = {}
        self.save_configs = {
            'model': 'models',
            'result': 'results',
        }
        train_tmp = pickle.load(open(os.path.join(root_dir, 'train.pkl'), 'rb'))
        val_tmp = pickle.load(open(os.path.join(root_dir, 'val.pkl'), 'rb'))
        test_tmp = pickle.load(open(os.path.join(root_dir, 'test.pkl'), 'rb'))
        volume_k = 2
        neighbor_index = list(range(-1 * volume_k, 0)) + list(range(1, volume_k + 1))
        self.dataset_configs = {
            'train': {'shuffle': True, 'transform': self._get_transform(),
                        'images': train_tmp, 'neigbor_index': neighbor_index, 'has_hr': True,
                        'image_root_dir': os.path.join(root_dir, 'images'),
                        'fill_method': 'LR'},
            'val': {'shuffle': False, 'transform': self._get_transform(),
                        'images': val_tmp, 'neigbor_index': neighbor_index, 'has_hr': True,
                        'image_root_dir': os.path.join(root_dir, 'images'), 
                        'fill_method': 'LR'},
            'test': {'shuffle': False, 'transform': self._get_transform(),
                        'images': test_tmp, 'neigbor_index': neighbor_index, 'has_hr': False,
                        'image_root_dir': os.path.join(root_dir, 'images'), 
                        'fill_method': 'LR'}
        }
        

    def get_configs_dict(self):
        configs = {
            'regular': self.regular_configs,
            'optimizer': self.optimizer_configs,
            'model': self.model_configs,
            'dataset': self.dataset_configs,
            'save': self.save_configs,
        }
        return configs

    def _get_transform(self):
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # return transform
        return my_transform
    
def my_transform(x):
    x = np.array(x, dtype='float32')
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)
    return x

