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

DATA_PATH = './data/'
dataset_orignial = load_data(DATA_PATH + 'original_target_data.npz')
x,y,test_x,test_y = dataset_orignial
del test_x,test_y







