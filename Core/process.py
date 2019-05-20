# -*- coding: UTF-8 -*-
import abc
import torch.optim as optim
from utils import get_SVR_loaders
import os

class ProcessBase:

    def init_parameters(self, model_class, dataset_class, configs):
        self.configs = configs
        self.model = model_class(self.configs.model_configs)
        self.device = self.configs.regular_config['device']
        self.model = self.model.to(self.device)
        self.data_loaders = get_SVR_loaders(dataset_class, 
            self.configs.regular_config, self.configs.data_set_config)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
            self.model.parameters()),
            lr=self.configs.optimizer_configs['lr'], 
            weight_decay=self.configs.optimizer_configs['weight_decay'])
        self.model_save_dir = self.configs.save_configs['model']
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.result_save_dir = self.configs.save_configs['result']
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