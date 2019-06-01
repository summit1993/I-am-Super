# -*- coding: UTF-8 -*-
import sys, os, pickle
from EDVR_config import EDVR_Configs
from EDVR_process import EDVR_Process

root_dir = '/data1/youku'

train_tmp = pickle.load(open(os.path.join(root_dir, 'pickle/train_val_crop.pkl'), 'rb'))
test_tmp = pickle.load(open(os.path.join(root_dir, 'pickle/test_crop.pkl'), 'rb'))
images_root_dir = os.path.join(root_dir, 'images_crop')
circle = 12
str_fill = 0

volume_k = 2
configs = EDVR_Configs(volume_k=volume_k)

configs.regular_configs['batch_size'] = 1
configs.regular_configs['show_iters'] = 1


configs.model_configs = {
    'model_name': 'EDVR_Patch',
    'nf': 128,
    'nframes': volume_k * 2 + 1,
    'groups': 8,
    'front_RBs': 5,
    'back_RBs': 40,
    'center': None,
    'predeblur': False,
    'HR_in': False,
    'add_padding': (2, 0),
}

configs.dataset_configs['train']['images'] = train_tmp
configs.dataset_configs['train']['image_root_dir'] = images_root_dir
configs.dataset_configs['train']['circle'] = circle
configs.dataset_configs['train']['str_fill'] = str_fill
configs.dataset_configs['test']['images'] = test_tmp
configs.dataset_configs['test']['image_root_dir'] = images_root_dir
configs.dataset_configs['test']['circle'] = circle
configs.dataset_configs['test']['str_fill'] = str_fill

configs.dataset_configs.pop('val')

process_model = EDVR_Process(configs)
process_model.process()