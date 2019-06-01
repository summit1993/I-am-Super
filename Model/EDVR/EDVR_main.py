# -*- coding: UTF-8 -*-
import sys, os, pickle
from EDVR_config import EDVR_Configs
from EDVR_process import EDVR_Process

# root_dir = '/data1/youku'
root_dir = 'H:\\data\\youku'

train_tmp = pickle.load(open(os.path.join(root_dir, 'train_val.pkl'), 'rb'))

configs = EDVR_Configs()

configs.regular_configs['batch_size'] = 1
configs.regular_configs['show_iters'] = 1

configs.dataset_configs['train']['images'] = train_tmp
# configs.dataset_configs.pop('train')
configs.dataset_configs.pop('val')

process_model = EDVR_Process(configs)
process_model.process()
