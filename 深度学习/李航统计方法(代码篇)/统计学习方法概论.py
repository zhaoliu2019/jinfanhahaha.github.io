# 如何用代码实现"用目标函数y=sin2πx, 加上一个正态分布的噪音干扰，用多项式去拟合"

# 首先得自己审题，自己思考一遍该如何用代码去实现
# 以下是我看过黄海广老师代码后的思路

# 一、根据题意，我们先写出目标函数的代码。
import numpy as np
def real_func(x) :
  return np.sin(2*np.pi*x)
# 二、有了函数，我们还需要一组x代入函数得到对应的y，所以接下来就是创建一组x,因为sin函数的周期是2π，所以我们取x的范围为（0，1）
x = np.linspace(0, 1, 10)
y_ = real_func(x) # 真实y
# 题目要求我们加上一个正态分布的噪音干扰
y = [np.random.normal(0, 0.1) + y1 for y1 in y_] # 噪音y
# 三、接下去，题目要求用多项式取拟合目标函数y=sin2πx，首先我们要有多项式的函数、
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x) # ps: numpy.poly1d([1,2,3]) 生成 1乘x的二次方+2乘x的一次方+3乘x的0次方
# 因为是拟合，肯定存在误差，即残差，我们把残差用代表表示出来很简单，即预测y - 真实y
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret
# 书中提到  高斯于1823年在误差e1,…,en独立同分布的假定下,证明了最小二乘方法的一个最优性质:（证明略）
# 在所有无偏的线性估计类中,最小二乘方法是其中方差最小的！ 对于数据(xi,yi)(i=1,2,3...,m)
# 所以我们用最小二乘法去求出多项式的各个参数，比较好的是scipy给我们封装好了最小二乘法leastsq()
# leastsq() 简单介绍：
# 它的参数很多，我们一般提取前三个参数 。func：误差函数 ，x0：表示函数的参数 ， args（）表示数据点
from scipy.optimize import leastsq
# 四、到此，我们可以自然的想到，写一个函数，它能够返回多项式的参数值，参数是经过最小二乘法最优的。
def fitting(M=0): # 加M参数目的：人为的控制多项式项数。
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0]) # 输出参数值
    return p_lsq
# 总感觉缺了什么？？？
# 哈撒ki ，原来是缺了可视化 ！！！ 一般来说我们在实现一个方案时，是需要可视化来进行评估和了解的 ，我们将可视化封装进fitting函数中
import matplotlib.pyplot as plt
def fitting(M=0): # 加M参数目的：人为的控制多项式项数。
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0]) # 输出参数值
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    return p_lsq

'''到此，就用代码实现了这个数学问题的思想，经过调用fitting方法测试不同的M值后，发现随着M的增大，咱们去拟合的线条越来越弯弯曲曲不规则，
   这个就是过拟合的表现，看过书的小伙伴肯定知道为啥会发生这个问题，解决这种问题的方法有啥。这个问题很明显就是参数过多，导致模型变复杂，发生了
   过拟合的现象，一般来说解决过拟合可以从以下几个方面入手：1、降低模型复杂度。2、正则化。3、增加样本数据。4、尝试化简，选择提取更好的特征。
   5、去噪。6、集成学习的方法。这里我们用正则化来实现降低模型复杂度的方法'''
# 正则化：给模型的参数加上了一定的正则约束，比如将权值的大小加入到损失函数中。
# 好，我们添加一个正则项（加上正则约束）
regularization = 0.0001
def residuals_func_regularization(p, x, y): 
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * regularization * np.square(p))) # L2范数作为正则化项
'''这是在原来的残差函数中，加上了L2正则化项后的函数，这一步的目的就是约束参数的大小，防止有些参数为了拟合数据而过大，导致过拟合。
   具体想深入了解，可以查阅正则化相关的内容'''
# 接下来就是将正则化封装进我们的函数里，并且实现可视化
def fitting(M=0):
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y)) # 正则约束
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))                               # 常规
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.plot(x_points,fit_func(p_lsq_regularization[0], x_points),label='regularization')
    plt.legend()
    return p_lsq
  


