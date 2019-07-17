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
 def fit(self, X_train, y_train, t0=5, t1=50):
   # 首先进行断言 判断用户输入的X_train和y_train是没有问题的
   assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
   
   # 学习率函数，越往后学习率应该越小，如果取固定值，到达最低点时很有可能又跳出最低点了，模拟了‘退火’思想
   def learning_rate(t):
     return t0 / (t + t1)
   
   # 原函数
   def sign(x, w, b):
     return np.dot(x, w) + b
   
   # 到此一切就绪，开始训练模型
   is_wrong = False
   while not is_wrong:
     wrong_count = 0
     for d in range(len(X_train)):
       X = X_train[d]
       y = y_train[d]
       if y * self.sign(X, self.w, self.b) <= 0:
         # 更新操作
         self.w = self.w + learning_rate(d) * np.dot(y, X)
         self.b = self.b + learning_rate(d) * y
         wrong_count += 1
     if wrong_count == 0:
       is_wrong = True
   return self
# 第三步，预测我们的测试集
def predict(self,X_predict):
   # 先断言，确保用户已经进行过fit操作
   assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
   # 确保测试集是有效的
   assert X_predict.shape[1] == len(self.w), \
            "the feature number of X_predict must be equal to X_train"
   y_predict = [self._predict(x) for x in X_predict]
   return np.array(y_predict)

# 构建私有的_predict方法用来对单个x进行预测
def _predict(self , x):
   # 断言，确保x是符合规范的
   assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
   if self.w.dot(x) + self.b >= 0 :
      return 1
   else:
      return -1
# 测试模型的准确度
def score(self, X_test, y_test):
   """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
   y_predict = self.predict(X_test)
   # 断言 确保输入的y_test符合规范
   assert len(y_test) == len(y_predict), \
        "the size of y_test must be equal to the size of y_predict"
   return np.sum(y_test == y_predict) / len(y_test)

  
  
