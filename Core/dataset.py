# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

L_size = (480, 270)
H_size = (1920, 1080)

class DatasetBase(Dataset):
    def __init__(self, param):
        self.image_root_dir = param['image_root_dir']
        self.image_list = param['images']
        self.has_hr = param['has_hr']
        self.neigbor_index = param['neigbor_index']
        self.neigbor_index.sort()
        self.transform = param['transform']
        self.fill_method = param['fill_method']
        self.image_fill_method = param['image_fill_method']
        self.circle = param['circle']
        self.str_fill = param['str_fill']

    def __len__(self):
        return len(self.image_list)

    def get_base_item(self, it):
        item = self.image_list[it]
        image_file = item['image']
        # image_index from 1, i.e. 1, 2, ...., n
        image_index = int(image_file.split('.')[0])
        LR_dir = os.path.join(self.image_root_dir, item['low_dir'])
        LR_image = self._read_image(os.path.join(LR_dir, image_file), L_size)
        LR_image_trans = self.transform(LR_image)
        neigbor_size = len(self.neigbor_index)
        HR_image = None
        if neigbor_size == 0:
            LR_neigbor = None
        else:
            LR_neigbor = torch.zeros(neigbor_size, LR_image_trans.shape[0],
                                LR_image_trans.shape[1], LR_image_trans.shape[2])
            for i in range(neigbor_size):
                index = self.neigbor_index[i] * self.circle
                if self.str_fill > 0:
                    image_path = os.path.join(LR_dir, str(image_index + index).zfill(self.str_fill) + '.bmp')
                else:
                    image_path = os.path.join(LR_dir, str(image_index + index) + '.bmp')
                if os.path.exists(image_path):
                    img_tmp = self._read_image(image_path, L_size)
                    img_tmp = self.transform(img_tmp)
                    LR_neigbor[i] = img_tmp
                else:
                    if self.fill_method == 'LR':
                        LR_neigbor[i] = LR_image_trans 

        if self.has_hr:
            HR_dir = os.path.join(self.image_root_dir, item['high_dir'])
            HR_image = self._read_image(os.path.join(HR_dir, image_file), H_size)
        
        return LR_image, LR_neigbor, HR_image

    def _read_image(self, image_path, right_size):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != right_size:
            if self.image_fill_method == 'padding' and img.size[0] < right_size[0] and img.size[1] < right_size[1]:
                img_numpy = np.array(img)
                img_large = np.zeros((right_size[1], right_size[0], 3), dtype='uint8')
                img_large[:img_numpy.shape[0],:img_numpy.shape[1]] = img_numpy
                img = Image.fromarray(img_large).convert('RGB')
            else:
                img = img.resize(right_size, Image.BICUBIC)
        return img

    def change_neibor_2_volume(self, LR, LR_neigbor):
        volume = torch.zeros(LR_neigbor.shape[0] + 1, LR_neigbor.shape[1],
                                LR_neigbor.shape[2], LR_neigbor.shape[3])
        index_0 = -1
        for i in range(len(self.neigbor_index)):
            if self.neigbor_index[i] > 0:
                index_0 = i
                break
        if index_0 == -1:
            index_0 = len(self.neigbor_index)
        volume[index_0] = LR
        if index_0 > 0:
            volume[:index_0] = LR_neigbor[:index_0]
        if index_0 < len(self.neigbor_index):
            volume[index_0 + 1:] = LR_neigbor[index_0:]
        return volume