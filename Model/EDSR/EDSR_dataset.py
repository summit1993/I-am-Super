# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.dataset import DatasetBase
from PIL import Image

class EDSR_Dataset(DatasetBase):
    def __init__(self, param):
        super(EDSR_Dataset, self).__init__(param)
    
    def __getitem__(self, it):
        LR_image, _, HR_image = self.get_base_item(it)
        LR = self.transform(LR_image)
        if HR_image is not None:
            HR = self.transform(HR_image)
        else:
            HR = it
        return (LR, HR)