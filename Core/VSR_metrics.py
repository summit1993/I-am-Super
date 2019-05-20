# -*- coding: UTF-8 -*
import numpy as np

def cal_img_PSNR(I,K):
    MSE = ((I - K)**2).mean()+1e-20
    PSNR = 20 * np.log10(255) - 10 * np.log10(MSE)
    return PSNR