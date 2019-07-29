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
w = tf.get_variable('w', [x.get_shape()[-1], 10],
                   initializer=tf.random_normal_initializer(0, 1))
# 再然后，设我们的偏置b,其中的initializer初始化为0.0的值
b = tf.get_variable('b', [10],
                   initializer=tf.constant_initializer(0.0))

# 根据神经元的公式，我们可以得到y=w·x+b
# [None, 3072] * [3072, 10] = [None, 10]
y_ = tf.matmul(x, w) + b

# [[0.01, 0.9, ..., 0.03], []]
p_y = tf.nn.softmax(y_)
# 5 -> [0,0,0,0,0,1,0,0,0,0]
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))

# 交叉熵的损失函数
# loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

'''我们也可以得到预测值'''
predict = tf.argmax(y_, 1)

'''我们还可以得到准确的那几个位置'''
correct_prediction = tf.equal(predict, y)


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
           all_data.append(data)
           all_labels.append(labels)
        self._data = np.vstack(all_data) # 将多个向量合并成矩阵
        self._data = self._data / 127.5 - 1  # 因为数据是像素，处于0-255之间，归一化使数值分布在-1～1之间
        self._labels = np.hstack(all_labels)  # 将向量变成多行一列的矩阵
        # 输出看一看我们的数据量
        print(self._data.shape)
        print(self._labels.shape)
        
        self._num_examples = self._data.shape[0] # 方便得到数据量
        self._need_shuffle = need_shuffle  # 将need_shuffle参数存入类中
        self._indicator = 0  # 起始位置（头指针），用来保存迭代到哪个位置
        
        # 如果传入参数need_shuffle为True，则执行打散操作
        if self._need_shuffle:
            self._shuffle_data()
          
    # 写私有的混乱函数
    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        # 获取混乱后的索引
        p = np.random.permutation(self._num_examples)
        # 更新
        self._data = self._data[p]
        self._labels = self._labels[p]
    
    # 写一个函数，获得下一批数据
    def next_batch(self, batch_size):
        # 计算尾指针的位置
        end_indicator = self._indicator + batch_size 
        # 如果尾指针大于数据量
        if end_indicator > self._num_examples:
            # 如果self._need_shuffle为True，进行打散操作，若不是，则报错
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0 # 将头指针更新为0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        # 若尾指针还比数据量大，则说明步长batch_size太大，进行报错
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        # 获取头指针和尾指针之间的数据
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        # 更新头指针
        self._indicator = end_indicator
        return batch_data, batch_labels

# 获取训练数据文件名称列表和测试数据文件名称列表
train_filenames = [os.path.join(CIFAR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR, 'test_batch')]

# 获取训练数据和测试数据
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False) # 测试集不需要打散

             '''数据到位，开始训练模型'''
# 初始化模型参数
init = tf.global_variables_initializer()
batch_size = 20 # 步长
train_steps = 10000 # 训练次数
test_steps = 100    # 测试次数，因为没有打散操作，次数有上限，因为这个测试集就10000个数据，所以不能超过500次

# 接下来，训练模型
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 循环输出
    for i in range(train_steps):
        # 得到其中一段训练数据和测试数据
        batch_data, batch_labels = train_data.next_batch(batch_size)
        # 调用run方法得到损失函数的值和精准率
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels})
        # 输出部分中间训练数据
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i+1, loss_val, acc_val))
        # 输出部分测试数据
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict = {
                        x: test_batch_data, 
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc))

            '''发现损失函数不同，会直接导致预测的准确率不同，以交叉熵为损失函数的预测值大概就是30%左右，
               而以均方误差的损失函数的准确率可以达到40%，对于多分类图片问题，我试了试传统的机器学习方法。'''
