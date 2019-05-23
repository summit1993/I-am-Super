# -*- coding: UTF-8 -*-
import abc
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import sys, os
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.utils import get_SVR_loaders
from Core.VSR_metrics import cal_img_PSNR

class ProcessBase:

    def init_parameters(self, model_class, dataset_class, configs):
        self.configs = configs
        self.model = model_class(self.configs.model_configs)
        if 'pre_model' in self.configs.model_configs:
            if torch.cuda.is_available():
                checkpoint = torch.load(self.configs.model_configs['pre_model'])
            else:
                checkpoint = torch.load(self.configs.model_configs['pre_model'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # self.model.train()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
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
            param['test_images'] = self.configs.dataset_configs['test']['images']
        for epoch in range(epoch_num):
            param['epoch'] = epoch
            self._train_process(param)
            self._val_process(param)
            self._test_process(param)
    
    def _train_process(self, param):
        if 'train_loader' in param:
            train_loader = param['train_loader']
        else:
            return
        self.model.train()
        criterion = param['criterion']
        show_iters = param['show_iters']
        epoch = param['epoch']
        model_save_epoch = param['model_save_epoch']
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            # we fix the data[-1] as label, i.e, HR image
            data = [a.to(self.device) for a in data]
            self.optimizer.zero_grad()
            outputs = self.model(data[:-1])
            loss = criterion(outputs, data[-1])
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
        print('*' * 10, 'Begin to validation', '*' * 10)
        with torch.no_grad():
            self.model.eval()
            count = 0.0
            PSNR = 0.0
            for _, val_data in enumerate(val_loader, 0):
                val_input = [a.to(self.device) for a in val_data[:-1]]
                outputs = self.model(val_input)
                outputs = outputs.to('cpu').numpy()
                outputs = outputs * 255
                outputs = np.rint(outputs)
                outputs[outputs < 0] = 0
                outputs[outputs > 255] = 255
                HR_images = val_data[-1].numpy()
                HR_images = HR_images * 255
                HR_images = np.rint(HR_images)
                HR_images[HR_images < 0] = 0
                HR_images[HR_images > 255] = 255
                for i  in range(HR_images.shape[0]):
                        PSNR += cal_img_PSNR(outputs[i], HR_images[i])
                count += HR_images.shape[0]
            PSNR = PSNR * 1.0 / count
            print('PSNR:\t', PSNR)
            fw = open(os.path.join(self.result_save_dir, 'val_result_' + str(epoch) + '.txt'), 'w')
            fw.write(str(PSNR))
            fw.close()
        print('*' * 10, 'Finish validation', '*' * 10)
    
    def _test_process(self, param):
        if 'test_loader' in param:
            test_loader = param['test_loader']
        else:
            return
        epoch = param['epoch']
        print('*' * 10, 'Begin to prediction', '*' * 10)
        with torch.no_grad():
            self.model.eval()
            test_images = param['test_images']
            for _, test_data in enumerate(test_loader, 0):
                test_input = [a.to(self.device) for a in test_data[:-1]]
                index_list = test_data[-1].numpy().astype('int')
                outputs = self.model(test_input)
                outputs = outputs.permute(0, 2, 3, 1)
                outputs = outputs.to('cpu').numpy()
                outputs = outputs * 255.0
                outputs = np.rint(outputs)
                outputs[outputs < 0] = 0
                outputs[outputs > 255] = 255
                for i in range(outputs.shape[0]):
                    image = Image.fromarray(outputs[i].astype('uint8')).convert('RGB')
                    item = test_images[index_list[i]]
                    Res_dir = os.path.join(self.result_save_dir, 
                        str(epoch), item['Res_dir'])
                    if not os.path.exists(Res_dir):
                        os.makedirs(Res_dir)
                    image.save(os.path.join(Res_dir, item['image']))
        print('*' * 10, 'Finish predaction', '*' * 10)