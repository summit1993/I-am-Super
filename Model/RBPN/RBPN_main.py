# -*- coding: UTF-8 -*-
from RBPN_config import RBPN_Configs
from RBPN_proces import RBPN_Process

configs = RBPN_Configs()
# configs.regular_configs['batch_size'] = 1
# configs.regular_configs['num_workers'] = 0
# configs.regular_configs['show_iters'] = 1

process_model = RBPN_Process(configs)
process_model.process()