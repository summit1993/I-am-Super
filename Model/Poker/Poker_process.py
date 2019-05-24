# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.process import ProcessBase
from Poker_model import Poker_Model
from Poker_dataset import Poker_Dataset

class Poker_Process(ProcessBase):
    def __init__(self, configs):
        super(Poker_Process, self).__init__()
        self.init_parameters(Poker_Model, Poker_Dataset, configs)