# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from Model.SISR.EDSR_model import EDSR_Model
from EDSR_dataset import EDSR_Dataset

class EDSR_Process(ProcessBase):
    def __init__(self, configs):
        super(EDSR_Process, self).__init__()
        self.init_parameters(EDSR_Model, EDSR_Dataset, configs)