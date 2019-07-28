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
# 再然后，设我们的偏置b,其中的initializer初始化为0.0的值
b = tf.get_variable('b', [1],
                   initializer=tf.constant_initializer(0.0))

# 根据神经元的公式，我们可以得到y=w·x+b
y_ = tf.matmul(x,w) + b  # 其中的matmul方法是矩阵相乘
# 将y_映射到sigmoid函数上
p_y_1 = tf.nn.sigmoid(y_)
# 因为y本来的是[1,n]维度的向量，要将它变形成[n,1]维度的矩阵
y_reshape = tf.reshape(y , (-1,1))
# 将y_reshape变成float32型的
y_reshaped_float = tf.cast(y_reshape, tf.float32)

'''到此，我们可以得到损失函数了  -> 用MSE表示'''
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

'''我们也可以得到预测值'''
predict = p_y_1 > 0.5

'''我们还可以得到准确的那几个位置'''
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshape)

'''由此，我们可以得到准确率'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

'''最后一步构建计算图用梯度下降'''
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

'''接下来写一个读取文件的函数'''
def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']
      
# 到此，我们写一个类来更好的控制我们的数据，以满足接下来训练的需求

class CifarData:
    def __init__(self, filenames, need_shuffle): # 初始化方法，need_shuffle参数是是否需要打乱我们的数据
        # 用来特征数据和存标签数据
        all_data = []
        all_labels = []
        # 循环读出文件夹下的文件
        for filename in filenames:
           data, labels = load_data(filename)
           for item, label in zip(data, labels):
               if label in [0, 1]:  # 取0和1的原因是咱们就只做二分类
                   all_data.append(item)
                   all_labels.append(label)
        self._data = np.vstack(all_data) # 将多个向量合并成矩阵
        self._data = self._data / 127.5 - 1  # 因为数据是像素，处于0-255之间，归一化使数值分布在-1～1之间
        self._labels = np.hstack(all_labels)  # 将向量变成多行一列的矩阵
        # 输出看一看我们的数据量
        print(self._data.shape)
        print(self._labels.shape)
        
        self._num_examples = self._data.shape[0] # 方便得到数据量
        self._need_shuffle = need_shuffle  # 将need_shuffle参数存入类中
        self._indicator = 0  # 
 
