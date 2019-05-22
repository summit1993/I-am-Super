# -*- coding: UTF-8 -*-
from FSTRN_config import FSTRN_Configs
from FSTRN_process import FSTRN_Process

configs = FSTRN_Configs()
configs.model_configs['type'] = 'poker'
configs.regular_configs['batch_size'] = 16
# configs.regular_configs['num_workers'] = 0
# configs.regular_configs['show_iters'] = 1

process_model = FSTRN_Process(configs)
process_model.process()