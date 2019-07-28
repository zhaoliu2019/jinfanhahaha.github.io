''' 首先第一步肯定是导入相应的包'''
import numpy as np
import pickle  # 读取网上的数据，那个格式就得用这个包来读取
import os
import tensorflow

'''接下去看看这个文件夹里都有啥文件'''
# 文件路径（个人看情况）
CIFAR = './cifar-10'
# 看一看都有啥文件
os.listdir(CIFAR)

# 以下是所有的文件，很明显，data_batch1-5都是训练集，而test_batch是测试集
'''['data_batch_1',
 'readme.html',
 'batches.meta',
 'data_batch_2',
 'data_batch_5',
 'test_batch',
 'data_batch_4',
 'data_batch_3']'''

'''接下来写一个读取文件的函数'''
def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']
      
'''
