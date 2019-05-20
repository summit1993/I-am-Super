# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import torch

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
            'lr': 1e-4,
            'weight_decay': 1e-5,
        }
        self.model_configs = {}
        self.dataset_configs = {}
        self.save_configs = {
            'model': 'models',
            'result': 'results',
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
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform

