# -*- coding: UTF-8 -*-
import sys, os, pickle
from Poker_config import Poker_Configs
from Poker_process import Poker_Process

# root_dir = '/data1/youku'
root_dir = 'H:\\data\\youku'

train_tmp = pickle.load(open(os.path.join(root_dir, 'train_val.pkl'), 'rb'))

configs = Poker_Configs()
configs.model_configs['type'] = 'poker'
configs.regular_configs['batch_size'] = 1
configs.regular_configs['num_workers'] = 0
configs.regular_configs['show_iters'] = 1

configs.dataset_configs['train']['images'] = train_tmp
# configs.dataset_configs.pop('train')
configs.dataset_configs.pop('val')

process_model = Poker_Process(configs)
process_model.process()