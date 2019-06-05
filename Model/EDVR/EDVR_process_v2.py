# -*- coding: UTF-8 -*-
import os, sys
import torch
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from Core.utils import get_SVR_loaders
from EDVR_model import EDVR_Model
from EDVR_dataset import EDVR_Dataset
import random

class EDVR_Process_V2(ProcessBase):
    def __init__(self, configs):
        super(EDVR_Process_V2, self).__init__()
        self.init_parameters(EDVR_Model, EDVR_Dataset, configs)
    
    def _train_process(self, param):
        if 'train' in self.configs.dataset_configs:
            train_configs = self.configs.dataset_configs['train']
        else:
            return
        
        criterion = param['criterion']
        show_iters = param['show_iters']
        epoch = param['epoch']
        model_save_epoch = param['model_save_epoch']
        running_loss = 0.0
        
        train_fold_configs = train_configs.copy()
        fold_configs = train_configs.copy()
        fold_configs['shuffle'] = False

        images = train_configs['images']
        random.shuffle(images) 
        folds_num = self.configs.model_configs['folds_num']
        fold_images_num = int(len(images) / folds_num)
        select_num = self.configs.model_configs['select_num']

        fold_images = [images[a] for a in random.sample(range(len(images)), select_num)]

        step_begin = 0

        for t in range(folds_num):
            train_fold_configs['images'] = fold_images
            train_fold_loader = get_SVR_loaders(EDVR_Dataset, 
                self.configs.regular_config, {'train': train_fold_configs})['train']
            self.model.train()
            for step, data in enumerate(train_fold_loader, 0):
                data = [a.to(self.device) for a in data]
                self.optimizer.zero_grad()
                outputs = self.model(data[:-1])
                loss = criterion(outputs, data[-1])
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if show_iters > 0:
                    if (step + step_begin) % show_iters == (show_iters - 1):
                        print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, step + step_begin +  1, running_loss / show_iters))
                        running_loss = 0.0

            # collect next fold images
            self.model.eval()
            fold_all_images = images[t * fold_images_num : min(images, (t + 1) * fold_images_num)]
            train_fold_configs['images'] = fold_all_images
            fold_loader = get_SVR_loaders(EDVR_Dataset, 
                self.configs.regular_config, {'train': train_fold_configs})['train']
            with torch.no_grad():
                pass