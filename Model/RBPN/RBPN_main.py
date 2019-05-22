# -*- coding: UTF-8 -*-
from RBPN_config import RBPN_Configs
from RBPN_proces import RBPN_Process
import pickle, os

# root_dir = '/data1/youku'
# root_dir = 'D:\\program\\deep_learning\\I-am-Super\\data'
root_dir = 'H:\\data\\youku'

train_tmp = pickle.load(open(os.path.join(root_dir, 'train_val.pkl'), 'rb'))

configs = RBPN_Configs()
configs.regular_configs['batch_size'] = 1
# configs.regular_configs['num_workers'] = 0
# configs.regular_configs['show_iters'] = 1

configs.dataset_configs['train']['images'] = train_tmp
configs.dataset_configs.pop('val')


process_model = RBPN_Process(configs)
process_model.process()