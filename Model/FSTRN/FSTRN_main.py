# -*- coding: UTF-8 -*-
from FSTRN_config import FSTRN_Configs
from FSTRN_process import FSTRN_Process

configs = FSTRN_Configs()

process_model = FSTRN_Process(configs)
process_model.process()