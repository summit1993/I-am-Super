# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from FSTRN_model import FSTRN_Model
from FSTRN_dataset import FSTRN_Dataset

class FSTRN_Process(ProcessBase):
    def __init__(self, configs):
        super(FSTRN_Process, self).__init__()
        self.init_parameters(FSTRN_Model, FSTRN_Dataset, configs)