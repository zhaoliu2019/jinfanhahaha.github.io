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

# 我们取出其中一个训练集，看看里面的内容
with open(os.path.join(CIFAR,'data_batch_1') , 'rb') as f:  # rb的意思就是将文件读成二进制格式
  # 取出data
  data = pickle.load(f,encoding='bytes')
  # 看一看data的类型
  print(type(data))
  '''<class 'dict'>'''
  # 看看都有哪些键
  print(data.keys())
  '''dict_keys([b'batch_label', b'labels', b'data', b'filenames'])'''
  # 根据这些键，来了解这些数据
  print(type(data[b'data']))
  print(type(data[b'batch_label']))
  print(type(data[b'labels']))
  print(type(data[b'filenames']))
  print(data[b'data'].shape)
  print(Counter(data[b'labels']))
  print(data[b'filenames'][1])
  print(data[b'data'][:2])
  print(data[b'labels'][:2])
  '''<class 'numpy.ndarray'>
  <class 'bytes'>
  <class 'list'>
  <class 'list'>
  (10000, 3072) 
  Counter({2: 1032, 6: 1030, 8: 1025, 3: 1016, 0: 1005, 7: 1001, 4: 999, 9: 981, 1: 974, 5: 937})
  b'camion_s_000148.png'
  [[ 59  43  50 ... 140  84  72]
   [154 126 105 ... 139 142 144]]
  [6, 9]'''
  
'''到此，有来数据，接下来，让我们来Hello CV吧，嘿嘿。'''
# 我对HelloCV的概念就是可视化图片 ， 我们先取出其中的一张图片，对它进行可视化
image_arr = data[b'data'][1]

'''已知它是3072个数据，在这里我解释以下3072，通俗的说就是32乘32乘3，其中32乘32是图片的像素，另一个3其实是R G B三种像素，
   在神经网络里表示三个通道，我们想还原出照片，首先就得将3072这个一维数组变成三维数组（32，32，3），但是这个数据本身其实
   是（3，32，32）的，所以运用numpy中对数组的操作，进行变形'''
image_arr = image_arr.reshape((3,32,32))
image_arr = image_arr.transpose((1,2,0)) # 到此 ，数组变成了我们想要的形式（32，32，3）了

# 导入画图工具
import matplotlib.pyplot as plt

# 直接调用matplotlib封装好的imshow函数，进行还原
from matplotlib.pyplot import imshow

'''若是在jupyter中进行画图，想要可视化还得在加这一行代码'''
%matplotlib inline

# Hello CV
imshow(image_arr)

  
