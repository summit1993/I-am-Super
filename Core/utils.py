# -*- coding: UTF-8 -*
from torch.utils.data import DataLoader
import os
from PIL import Image

def get_SVR_loaders(SVRDataset, regular_config, data_set_config):
    loader_dict = {}
    for key in ['train', 'val', 'test']:
        if key not in data_set_config:
            continue
        value = data_set_config[key]
        loader_set = SVRDataset(value)
        loader_dict[key] = DataLoader(loader_set, batch_size=regular_config['batch_size'],
                                      shuffle=value['shuffle'],
                                      num_workers=regular_config['num_workers'])
    return loader_dict


def crop_patch_from_image(img_root_dir, save_root_dir, img_shape=(480, 270), patch_nums=(4, 3)):
    dirs = os.listdir(img_root_dir)
    img_stride = patch_nums[0] * patch_nums[1]
    x_stride = img_shape[0] / patch_nums[0]
    y_stride = img_shape[1] / patch_nums[1]
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    for dir in dirs:
        dir_tmp = os.path.join(img_root_dir, dir)
        if os.path.isdir(dir_tmp):
            imgs = os.listdir(dir_tmp)
            save_dir = os.path.join(save_root_dir, dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for img_name in imgs:
                img_index = (int(img_name.split('.')[0]) - 1) * img_stride
                img_file = os.path.join(dir_tmp, img_name)
                img = Image.open(img_file)
                if img.size != img_shape:
                    img = img.resize(img_shape, Image.BICUBIC)
                for i in range(patch_nums[0]):
                    for j in range(patch_nums[1]):
                        t = img_index + i * patch_nums[1] +  j + 1
                        patch = img.crop((i * x_stride, j * y_stride, (i + 1) * x_stride, (j + 1) * y_stride))
                        patch.save(os.path.join(save_dir, str(t) + '.bmp'), 'bmp')

def crop_patch_from_image_v2(img_root_dir, save_root_dir, img_shape=(480, 270), patch_nums=(10, 5), patch_shape=(64, 64)):
    dirs = os.listdir(img_root_dir)
    img_stride = patch_nums[0] * patch_nums[1]
    x_stride = img_shape[0] / patch_nums[0]
    x_pad = int((patch_shape[0] - x_stride) / 2)
    y_stride = img_shape[1] / patch_nums[1]
    y_pad = int((patch_shape[1] - y_stride) / 2)
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    for dir in dirs:
        dir_tmp = os.path.join(img_root_dir, dir)
        if os.path.isdir(dir_tmp):
            imgs = os.listdir(dir_tmp)
            save_dir = os.path.join(save_root_dir, dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for img_name in imgs:
                img_index = (int(img_name.split('.')[0]) - 1) * img_stride
                img_file = os.path.join(dir_tmp, img_name)
                img = Image.open(img_file)
                if img.size != img_shape:
                    img = img.resize(img_shape, Image.BICUBIC)
                for i in range(patch_nums[0]):
                    for j in range(patch_nums[1]):
                        t = img_index + i * patch_nums[1] +  j + 1
                        crop_begin_x = i * x_stride - x_pad
                        crop_end_x = (i + 1) * x_stride + x_pad
                        crop_begin_y = i * y_stride - y_pad
                        crop_end_y = (i + 1) * y_stride + y_pad
                        if crop_begin_x < 0:
                            crop_begin_x = 0
                            crop_end_x = patch_shape[0]
                        if crop_end_x > img_shape[0]:
                            crop_begin_x = img_shape[0] - patch_shape[0]
                            crop_end_x = img_shape[0]
                        if crop_begin_y < 0:
                            crop_begin_y = 0
                            crop_end_y = patch_shape[1]
                        if crop_end_y > img_shape[1]:
                            crop_begin_y = img_shape[1] - patch_shape[1]
                            crop_end_y = img_shape[1]
                        patch = img.crop((crop_begin_x, crop_begin_y, crop_end_x, crop_end_y))
                        patch.save(os.path.join(save_dir, str(t) + '.bmp'), 'bmp')
