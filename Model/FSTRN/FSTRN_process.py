# -*- coding: UTF-8 -*-
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.utils import get_SVR_loaders
from Core.VSR_metrics import cal_img_PSNR
from Core.process import ProcessBase
from FSTRN_model import FSTRN_Model
from FSTRN_dataset import FSTRN_Dataset

class FSTRN_Process(ProcessBase):
    def __init__(self, configs):
        super(FSTRN_Process, self).__init__()
        self.init_parameters(FSTRN_Model, FSTRN_Dataset, configs)

    def process(self):
        param = {}
        epoch_num = self.configs.regular_configs['epoch_num']
        param['show_iters'] = self.configs.regular_configs['show_iters']
        param['model_save_epoch'] = self.configs.regular_configs['model_save_epoch']
        param['criterion'] = nn.L1Loss()
        if 'train' in self.data_loaders:
            param['train_loader'] = self.data_loaders['train']
        if 'val' in self.data_loaders:
            param['val_loader'] = self.data_loaders['val']
        if 'test' in self.data_loaders:
            param['test_loader'] = self.data_loaders['test']
        for epoch in range(epoch_num):
            param['epoch'] = epoch
            self._train_process(param)
            self._val_process(param)
    
    def _train_process(self, param):
        self.model.train()
        criterion = param['criterion']
        show_iters = param['show_iters']
        train_loader = param['train_loader']
        epoch = param['epoch']
        model_save_epoch = param['model_save_epoch']
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            LR_volums, HR_images, LR_R_image = data
            LR_volums, HR_images, LR_R_image = LR_volums.to(self.device), HR_images.to(self.device), LR_R_image.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model([LR_volums, LR_R_image])
            loss = criterion(outputs, HR_images)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if show_iters > 0:
                if step % show_iters == (show_iters - 1):
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, running_loss / show_iters))
                    running_loss = 0.0

            if model_save_epoch > 0:
                if (epoch + 1) % model_save_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.model_save_dir, 'checkpoint_' + str(epoch) + '.tar'))
        
    def _val_process(self, param):
        if 'val_loader' in param:
            val_loader = param['val_loader']
        else:
            return
        epoch = param['epoch']
        with torch.no_grad():
            self.model.eval()
            print('*' * 10, 'Begin to validation', '*' * 10)
            count = 0.0
            PSNR = 0.0
            for _, val_data in enumerate(val_loader, 0):
                LR_volums, HR_images, LR_R_image = val_data
                LR_volums, LR_R_image = LR_volums.to(self.device), LR_R_image.to(self.device)
                outputs = self.model([LR_volums, LR_R_image])
                outputs = outputs.to('cpu').numpy()
                outputs = np.rint(outputs)
                outputs[outputs < 0] = 0
                outputs[outputs > 255] = 255
                HR_images = HR_images.numpy()
                for i  in range(HR_images.shape[0]):
                        PSNR += cal_img_PSNR(outputs[i], HR_images[i])
                count += HR_images.shape[0]
            PSNR = PSNR * 1.0 / count
            print('PSNR:\t', PSNR)
            fw = open(os.path.join(self.result_save_dir, 'val_result_' + str(epoch) + '.txt'), 'w')
            fw.write(str(PSNR))
            fw.close()
            print('*' * 10, 'Finish validation', '*' * 10)
    
    def _test_process(self):
        pass
