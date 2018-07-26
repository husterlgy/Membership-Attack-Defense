# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:04:41 2018

@author: husterlgy
python 2.7
xgboost 0.72
numpy 1.14.4
theano 1.0.1

"""
from attack import train_target_model
from attack import train_shadow_models
from attack import train_attack_model
from attack import load_data
from attack import save_data
from classifier import train as train_model, iterate_minibatches, load_dataset
import theano.tensor as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
from sklearn import preprocessing
import theano
import lasagne
import argparse
import imp
import os
import numpy as np
import pandas as pd 
import xgboost as xgb


np.random.seed(21312)
MODEL_PATH = './model/'
DATA_PATH = './data/'
ORIG_DATA_PATH = './UCI_Adult_data/'
NOISE_DATA_PATH = './data/noised_data/'
NOISE_MODEL_PATH = './model/noised_model/'

import theano.gof.compiledir as cd
cd.print_compiledir_content()

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    
if not os.path.exists(NOISE_DATA_PATH):
    os.makedirs(NOISE_DATA_PATH)
    
if not os.path.exists(NOISE_MODEL_PATH):
    os.makedirs(NOISE_MODEL_PATH)

def coder_columns(columns,df_date):#把数据中的那些非数值型的变量全部转化成为数值型

    coders = []
    for name in columns:
        coder = preprocessing.LabelEncoder()
        coder.fit(df_date[name].values)
        df_date[name] = coder.transform(list(df_date[name].values))
        coders.append(coder)
    return coders,df_date

if __name__ =="__main__":
    '''Part-1：UCI前期数据处理'''
    if(os.path.exists(DATA_PATH + 'target_data.npz') !=True):
        print('File target_data.npz does not exist!')
        Train_Data_Set = pd.read_csv(ORIG_DATA_PATH+"/adult.data",sep=',', header=None, index_col=False)
        Test_Data_Set = pd.read_csv(ORIG_DATA_PATH+"/adult.test", sep=',', header=None, index_col=False)
    
    
        pd.set_option("display.width",400)
    
        # Naming the columns :
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                   'hours-per-week', 'native-country','income']
    
        Train_Data_Set.columns = columns
        Test_Data_Set.columns = columns
    
        Train_Data_Set["set"] = "train"
        Test_Data_Set["set"] =  "test"
        
        df_date = Train_Data_Set.append(Test_Data_Set,ignore_index=True)
        
        #对x数值进行编码
        coders,Train_Data_Set= coder_columns(["workclass","education","marital-status","occupation",
                                                   "relationship","race","sex","native-country"],df_date)
        
        
        Train_Data_Set = df_date[df_date["set"] == "train"].reset_index(drop=True)
        Test_Data_Set = df_date[df_date["set"] == "test"].reset_index(drop=True)
        
        print (Train_Data_Set[:10])
        workclasssum = Train_Data_Set.groupby("workclass").size()
        #对y 进行编码
        print(Train_Data_Set[:10])
        coder = preprocessing.LabelEncoder()
        coder.fit(Train_Data_Set["income"])
        Train_Data_Set["income"] = coder.transform(list(Train_Data_Set["income"].values))
        print (Train_Data_Set[:10])
        
        print(Test_Data_Set[:10])
        coder = preprocessing.LabelEncoder()
        coder.fit(Test_Data_Set["income"])
        Test_Data_Set["income"] = coder.transform(list(Test_Data_Set["income"].values))
        print(Test_Data_Set[:10])
        
        del df_date
        df_date = Train_Data_Set.append(Test_Data_Set,ignore_index=False)
        df_date = df_date.drop(["set"],axis=1)
        
    #    Test_Data_Set = Test_Data_Set.drop(["set"],axis=1)
    #    Train_Data_Set = Train_Data_Set.drop(["set"],axis=1)
    #    y_train = Train_Data_Set.pop("income")
    #    y_test = Test_Data_Set.pop("income")
    #    Test_Data_Set.to_csv(DATA_PATH+'test_data',index=False,header = False,sep=',')
    #    Train_Data_Set.to_csv(DATA_PATH+'train_data',index=False,header = False,sep=',')
    #    y_train.to_csv(DATA_PATH+'y_train',index=False,header = False)
    #    y_test.to_csv(DATA_PATH+'y_test',index=False,header = False)
        '''数据处理以接入attacker'''
        x = df_date
        y = x.pop("income")
        x = np.array(x)
        y = np.array(y)
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        test_x = None
        test_y = None
        '''这里有一个非常重要的地方，就theano只能使用32bit的数据，必须将数据强制转换成为32比特的'''
        np.savez(DATA_PATH + 'original_target_data.npz', x, y,test_x,test_x)
        dataset_orignial = load_data(DATA_PATH + 'original_target_data.npz')
    elif(os.path.exists(DATA_PATH + 'target_data.npz') ==True):
        '''数据已经处理好，接下来用到attacker中'''
        print('File original_target_data.npz is already existed!')
        dataset_orignial = load_data(DATA_PATH + 'original_target_data.npz')
            
    
    
    '''Part-2：攻击检测的相关变量初始化，其中初始化是否要保存数据到本地，是否添加噪声等等'''
    parser = argparse.ArgumentParser()

    # target and shadow model configuration    
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    parser.add_argument('--data_noised', type=int, default=0)   #   用来标记原始数据是否受到noise
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.3)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))   # number of data point used in target model
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=100)
    parser.add_argument('--target_n_hidden', type=int, default=50)
    parser.add_argument('--target_epochs', type=int, default=50)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-6)

    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='softmax')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=50)
    parser.add_argument('--attack_epochs', type=int, default=50)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)
    parser.add_argument('--DATA_PATH', type=str, default='./data/')
    parser.add_argument('--MODEL_PATH', type=str, default='./model/') 
    # parse configuration
    args = parser.parse_args()
    

        
    
    '''Part-3：进行攻击''' 
    if args.data_noised == 1:#如果数据被噪声污染，则切换数据目录
        args.DATA_PATH = './data/noised_data/'
        args.MODEL_PATH = './model/noised_model/'  
        ''''''''''''''''''''''''''''''''''''
        '''这里添加用来混淆数据的代码Start Coding'''''
        #  从这里开始为数据添加噪声
        
            
        
    
    
    
    
    
        '''这里添加用来混淆数据的代码End Coding'''''
        ''''''''''''''''''''''''''''''''''''
                
        
        
    if args.save_data:
        save_data(dataset_orignial,args,DATA_PATH = args.DATA_PATH, MODEL_PATH = args.MODEL_PATH)
        
        
        
        

    #所有的数据都是提前分配处理好，然后保存到本地的硬盘当中。
    #后续所有的训练操作都是从本地文件中读取数据，再进行训练。
    dataset = load_data(args.DATA_PATH + 'target_data.npz')
    print('Loading File From '+ args.DATA_PATH)
    print '-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n'
    attack_test_x, attack_test_y, test_classes = train_target_model(
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model,
        DATA_PATH = args.DATA_PATH,
        MODEL_PATH = args.MODEL_PATH)

    print '-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n'
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model,
        DATA_PATH = args.DATA_PATH,
        MODEL_PATH = args.MODEL_PATH)
    
    
    print '-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n'
    if args.data_noised == 1:
        print('-'*10 + 'Data is Noised!' + '-'*10)
    elif args.data_noised == 0:
        print('-'*10 + 'Data is Pure!' + '-'*10)
        
    dataset_for_attack = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    train_attack_model(
        dataset=dataset_for_attack,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes),
        DATA_PATH = args.DATA_PATH,
        MODEL_PATH = args.MODEL_PATH)




















