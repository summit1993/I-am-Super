# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from EDVR_model import EDVR_Model
from EDVR_dataset import EDVR_Dataset

class EDVR_Process(ProcessBase):
    def __init__(self, configs):
        super(EDVR_Process, self).__init__()
        self.init_parameters(EDVR_Model, EDVR_Dataset, configs)
        