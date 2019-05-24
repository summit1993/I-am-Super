# -*- coding: UTF-8 -*-
import os, sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from Core.dataset import DatasetBase
from PIL import Image

class Poker_Dataset(DatasetBase):
    def __init__(self, param):
        super(Poker_Dataset, self).__init__(param)

    def __getitem__(self, it):
        LR_image, LR_neigbor, HR_image = self.get_base_item(it)
        LR = self.transform(LR_image)
        if HR_image is not None:
            HR = self.transform(HR_image)
        else:
            HR = it
        LR_volume = self.change_neibor_2_volume(LR, LR_neigbor)
        LR_volume = LR_volume.permute(1, 0, 2, 3)
        return (LR_volume, LR, HR)