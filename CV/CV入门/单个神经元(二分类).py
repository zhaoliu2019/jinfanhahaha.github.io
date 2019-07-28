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
test_steps = 100    # 测试次数，因为没有打散操作，次数有上限，因为这个测试集就2000个数据，所以不能超过100次

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


''' 到此，我们就用一个用了逻辑斯蒂的神经元实现了图片二分类，准确率可以达到82%左右'''
# 换个角度想一想，图片分类能用机器学习吗？抱着试一试的态度，我尝试了以下LogisticRegression，SVC，KNeighborsClassifier
'''LogisticRegression的准确率可以达到81%多'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2' , C = 0.5,solver='sag')
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

'''惊讶的来了，我用SVC训练的准确度可以达到86.75%，原理分类图片不一定要用神经网络呢，如果只是二分类图片，
   传统的机器学习方法并不差，但是图片分类往往是多分类，这个时候几句要用神经网络了.'''
from sklearn.svm import SVC
svc = SVC(C=0.5)
svc.fit(X_train,y_train)
svc.score(X_test,y_test)

'''又试了试KNN，发现KNN分类图片效果不是特别好，只有63.4%的准确率'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

'''想到二分类图片，传统的机器学习方法不错，试了试集成学习的方法，首先登场的是VotingClassifier'''
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()), 
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],
                             voting='soft')
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

'''接下来登场是bagging中的oob方法'''
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                               n_estimators=50, max_samples=10,
                               bootstrap=True, oob_score=True)
bagging_clf.fit(X_train, y_train)
bagging_clf.oob_score_

'''AdaBoostClassifier也得看看'''
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
ada_clf.score(X_test, y_test)

'''最后在看一看GradientBoostingClassifier'''
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
gb_clf.score(X_test, y_test)

'''奈何数据量有点多，跑的有点慢，代码在这了。拜了个拜'''


