# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.dataset import DatasetBase
from PIL import Image

class EDVR_Dataset(DatasetBase):
    def __init__(self, args):
        super(EDVR_Dataset, self).__init__(args)
    
    def __getitem__(self, it):
        LR_image, LR_neigbor, HR_image = self.get_base_item(it)
        LR = self.transform(LR_image)
        if HR_image is not None:
            HR = self.transform(HR_image)
        else:
            HR = it
        LR_volume = self.change_neibor_2_volume(LR, LR_neigbor)
        return (LR_volume, HR)
    
