import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.model.handler import train, test
import argparse
from data_loader.SiteBinding_dataloader0 import *
import numpy as np
import pandas as pd
# from .models.Utils import *
from models.Utils import *

import random

# # 设置随机种子，保证训练过程中结果的一致性
# seed = 1  # 你可以选择任何整数作为种子
#
# # 设置PyTorch的随机种子
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
# # 设置Python的随机种子
# random.seed(seed)
#
# # 设置NumPy的随机种子
# np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='m6A_data')
parser.add_argument('--window_size', type=int, default=1)
parser.add_argument('--horizon', type=int, default=0)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3) #1e-3
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=5)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=20)
parser.add_argument('--decay_rate', type=float, default=0.5) #0.5
parser.add_argument('--dropout_rate', type=float, default=0.8) #0.5
parser.add_argument('--leakyrelu_rate', type=int, default=0.5) #0.2
parser.add_argument('--seq_len', type=int, default=41) #0.2
parser.add_argument('--seq_len1', type=int, default=64) #0.2
parser.add_argument('--cluster_num', type=int, default=6) #将聚类数量定为最大值41？

parser.add_argument('--size', type=int, default=41)
parser.add_argument('--num', type=int, default=1)

# torch.cuda.set_device(0)

args = parser.parse_args()
print(f'Training configs: {args}')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 'Human_Brain','Mouse_brain','Mouse_heart','Human_Liver',
    #                  'Mouse_kidney', 'Mouse_liver', 'Mouse_test', 'rat_brain', 'rat_kidney','rat_liver'
    # 'Mouse_kidney', 'Mouse_liver', 'Mouse_test'
    seq_types = ['Mouse_heart']
    # seq_types =['rat_liver']
    if args.train:
        j = 0
        while j<len(seq_types):
            print(str(seq_types[j])+'_train_validation beging!')

            result_train_file = os.path.join('output', args.dataset, seq_types[j])
            result_test_file = os.path.join('output', args.dataset, 'test')
            if not os.path.exists(result_train_file):
                os.makedirs(result_train_file)
            if not os.path.exists(result_test_file):
                os.makedirs(result_test_file)

            if args.train: #训练加验证
                try:
                    before_train = datetime.now().timestamp()
                    i=0
                    all_result=[]
                    while i<3 :
                        if i!=1:
                        # if 1==1:
                            print('fold '+str(i)+' ')
                            print('-'*99)
                            trainData=np.load('./Pre-Encoding/data_Elmo4/'+str(seq_types[j])+'/Train_Test/all/TrainData'+str(i)+'.npy',allow_pickle=True).tolist() #数组中的存储顺序为：二级结构图、onehot标签、一级结构图、公共特征
                            testData=np.load('./Pre-Encoding/data_Elmo4/'+str(seq_types[j])+'/Train_Test/all/TestData'+str(i)+'.npy',allow_pickle=True).tolist()

                            # print(trainData[2][0])

                            args.batch_size=len(testData[0])
                            args.batch_size1=int(len(trainData[0])-3*args.batch_size-1)

                            print('Train begining!')

                            args.batch_size = len(testData[0])
                            print(args.batch_size)
                            print('Train begining!')
                            forecast_feature, result = train(trainData, testData, args, result_train_file, i)
                            all_result.append(result)

                            # forecast_feature,result=train(trainData, testData, args, result_train_file,i,seq_types[j])
                            # all_result.append(result)
                            StorFile(all_result, './Pre-Encoding/data_Elmo4/'+str(seq_types[j])+'/Result/result'+str(i)+'.csv')
                        i+=1
                    # StorFile(all_result, './Pre-Encoding/data/'+str(seq_types[j])+'/Result/result'+str(i)+'.csv')
                    after_train = datetime.now().timestamp()
                    print(f'Training took {(after_train - before_train) / 60} minutes')

                except KeyboardInterrupt:
                    print('-' * 99)
                    print('Exiting from training early')
            # if args.evaluate:
            #     before_evaluation = datetime.now().timestamp()
            #     test(test_data, args, result_train_file, result_test_file)
            #     after_evaluation = datetime.now().timestamp()
            #     print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

            print(str(seq_types[j]) + '_train_validation done!')
            j+=1

    if args.evaluate:
        print(args.evaluate)
