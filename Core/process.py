# -*- coding: UTF-8 -*-
import abc
import torch.optim as optim
import torch
import sys, os
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.utils import get_SVR_loaders
import os

class ProcessBase:

    def init_parameters(self, model_class, dataset_class, configs):
        self.configs = configs
        self.model = model_class(self.configs.model_configs)
        if 'pre_model' in self.configs.model_configs:
            checkpoint = torch.load(self.configs.model_configs['pre_model'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.train()
        self.device = self.configs.regular_configs['device']
        self.model = self.model.to(self.device)
        self.data_loaders = get_SVR_loaders(dataset_class, 
            self.configs.regular_configs, self.configs.dataset_configs)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
            self.model.parameters()),
            lr=self.configs.optimizer_configs['lr'], 
            weight_decay=self.configs.optimizer_configs['weight_decay'])
        self.model_save_dir = self.configs.save_configs['model']
        self.model_save_dir = os.path.join(self.model_save_dir, self.configs.model_configs['model_name'])
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.result_save_dir = self.configs.save_configs['result']
        self.result_save_dir = os.path.join(self.result_save_dir, self.configs.model_configs['model_name'])
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
          
    @abc.abstractmethod
    def process(self):
        pass

    @abc.abstractmethod
    def _train_process(self, param):
        pass

    @abc.abstractmethod
    def _val_process(self, param):
        pass

    @abc.abstractmethod
    def _test_process(self, param):
        pass