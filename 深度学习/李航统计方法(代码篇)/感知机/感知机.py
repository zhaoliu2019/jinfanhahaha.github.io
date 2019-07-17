'''理解感知机的思想后，如何用代码实现一个我们自己的感知机，网上有很多实现模型的框架，这里我们用scikit—learn框架模板来实现。
   模板如下：'''

class X:
  def __init__(self): # 初始化
    pass
  def fit(self):      # 训练
    pass
  def predict(self):  # 预测值
    pass
  def score(self):    # 准确率
    pass
  
# 开始表演

# 导入numpy库
import numpy as np

# 第一步 __init__方法，我们的感知机参数有系数向量w和截距b，进行初始化
def __init__(self):
  self.w = None
  self.b = None
  
# 第二步，用随机梯度来训练我们的模型
'''为了更好的理解随机梯度的代码，我先写个模板。'''
 def fit(self, X_train, y_train, n_iters=50, t0=5, t1=50):
   # 首先进行断言 判断用户输入的X_train和y_train是没有问题的
   assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
   # n_iters是随机的次数，因为只随机一次的效果往往不理想（比较批量梯度下降法次数算很少了）初始值为50，用户可根据需要进行改变
   assert n_iters >= 1
   
   # 原函数
   def J() :
      pass
   
   # 原函数的导数
   def dJ（）:
      pass
   
   # 学习率函数，越往后学习率应该越小，如果取固定值，到达最低点时很有可能又跳出最低点了，模拟了‘退火’思想
   def learning_rate(t):
     return t0 / (t + t1)

   # initial_theta的作用随机取出一列进行训练，确保随机梯度的随机性
   initial_theta = np.random.randn(X_b.shape[1])
   
   # 进行随机梯度的训练，
   def sgd(X_b, y, initial_theta, n_iters=5):
      '''...'''
      return w,b
   
   self.w , self.b = sgd(X_b, y_train, initial_theta, n_iters)

   return self

# 看过模板之后 再实现感知机的随机梯度就很简单了
 def fit(self, X_train, y_train, n_iters=50, t0=5, t1=50):
   # 首先进行断言 判断用户输入的X_train和y_train是没有问题的
   assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
   # n_iters是随机的次数，因为只随机一次的效果往往不理想（比较批量梯度下降法次数算很少了）初始值为50，用户可根据需要进行改变
   assert n_iters >= 1
   
   # 原函数的导数
   def dJ（）:
      



  
  
