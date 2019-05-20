# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

L_size = (480, 270)
H_size = (1920, 1080)

class FSTRN_Dataset(Dataset):
    def __init__(self, param):
        self.image_root_dir = param['image_root_dir']
        self.image_list = param['images']
        self.has_hr = param['has_hr']
        self.volume_k = param['volume_k']
        self.transform = param['transform']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, it):
        item = self.image_list[it]
        image_file = item['image']
        # image_index from 1, i.e. 1, 2, ...., n
        image_index = int(image_file.split('.')[0])
        LR_dir = os.path.join(self.image_root_dir, item['low_dir'])
        LR_image = self._read_image(os.path.join(LR_dir, image_file), L_size)
        LR_R_image = LR_image.resize(H_size, Image.BICUBIC)
        LR_R_image = self.transform(LR_R_image)
        LR_image = self.transform(LR_image)
        # get LR Volume, volume image num: (volume_k * 2 + 1)
        LR_Volume = torch.zeros(self.volume_k * 2 + 1, LR_image.shape[0],
                                LR_image.shape[1], LR_image.shape[2])
        HR_image = None
        if self.volume_k == 0:
            LR_Volume[0] = LR_image
        else:
            LR_Volume[self.volume_k] = LR_image
            left_begin = max(1, image_index - self.volume_k)
            right_end = min(image_index + self.volume_k, item['nums'])
            for k in range(image_index - left_begin):
                img_tmp = self._read_image(os.path.join(LR_dir, str(image_index - k - 1).zfill(3) + '.bmp'), L_size)
                img_tmp = self.transform(img_tmp)
                LR_Volume[self.volume_k - k - 1] = img_tmp
            for k in range(right_end - image_index):
                img_tmp = self._read_image(os.path.join(LR_dir, str(image_index + k + 1).zfill(3) + '.bmp'), L_size)
                img_tmp = self.transform(img_tmp)
                LR_Volume[self.volume_k + k + 1] = img_tmp
        LR_Volume = LR_Volume.permute(1, 0, 2, 3)
        if self.has_hr:
            HR_dir = os.path.join(self.image_root_dir, item['high_dir'])
            HR_image = self._read_image(os.path.join(HR_dir, image_file), H_size)
            HR_image = self.transform(HR_image)
        return LR_Volume, HR_image, LR_R_image

    def _read_image(self, image_path, right_size):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != right_size:
            img = img.resize(right_size, Image.BICUBIC)
        return img