# -*- coding: UTF-8 -*-
from FSTRN_config import FSTRN_Configs
from FSTRN_process import FSTRN_Process
import pickle, os

# root_dir = '/data1/youku'
# root_dir = 'D:\\program\\deep_learning\\I-am-Super\\data'
root_dir = 'H:\\data\\youku'

train_tmp = pickle.load(open(os.path.join(root_dir, 'train_val.pkl'), 'rb'))

configs = FSTRN_Configs()
configs.model_configs['type'] = 'poker'
configs.regular_configs['batch_size'] = 1
configs.regular_configs['num_workers'] = 0
configs.regular_configs['show_iters'] = 1

# configs.model_configs['pre_model'] = './models/FSTRN/checkpoint_0.tar'

configs.dataset_configs['train']['images'] = train_tmp
# configs.dataset_configs.pop('train')
configs.dataset_configs.pop('val')

process_model = FSTRN_Process(configs)
process_model.process()