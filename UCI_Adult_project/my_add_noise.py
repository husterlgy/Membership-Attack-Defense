# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:30:08 2018

@author: husterlgy
为原始数据添加噪声，在保证target model 的识别准确率的情况下，降低attack model 的准确率。
"""
import numpy as np
import pandas as pd
from attack import load_data
from attack import save_data


def laplaceMechanismCount(loc,scale=1):
    #loc, scale = 100., 1.
    s = np.random.laplace(loc, scale, 1)
    ss=s[0]
    ss = round(ss, 1)
    return ss
    
def add_noise(x,noise_level):
    #这里的x是将train test set整合后的整个的X数据集。 type=np.int32
    x_noised = np.zeros(x.shape)
    noise_level_for_iter = np.ones(x.shape[0])
    noise_level_for_iter = noise_level_for_iter*noise_level
    
    for ii in range(x.shape[1]):
        x_noised[:,ii] = map(laplaceMechanismCount, x[:,ii],noise_level_for_iter)
        
    return x_noised
        
    
























