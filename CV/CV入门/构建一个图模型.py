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

'''好，完成了第一步后，我们开始第二步，用tensorflow来构建我们模型图'''
# 设我们的x，placeholder方法相当于提供了一个占位符
x = tf.placeholder(tf.float32, [None, 3072])
# 设我们的y
y = tf.placeholder(tf.int64, [None])
# 接下去，设我们的权值w ， 其中initializer给它加了一个均值为0方差为1的正态分布
w = tf.get_variable('w', [x.get_shape()[-1], 1],
                   initializer=tf.random_normal_initializer(0, 1))
# 再然后，设我们的偏置b
b = tf.get_variable('b', [1],
                   initializer=tf.constant_initializer(0.0))
