# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from RBPN_model import RBPN_Model
from RBPN_dataset import RBPN_Dataset

class RBPN_Process(ProcessBase):
    def __init__(self, configs):
        super(RBPN_Process, self).__init__()
        self.init_parameters(RBPN_Model, RBPN_Dataset, configs)