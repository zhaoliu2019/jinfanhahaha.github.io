# SVM 支持向量机
**思想：找到泛化能力最好的决策边界，距离类别的最近样本最远**
# 线性问题
## Hard Margin SVM -- 适用于线性可分问题
### 点到直线的距离，在二维中为$\frac{\left | Ax+By+C \right |}{\sqrt{A^2+B^2}}$  推理到n维$\frac{\left | w^{T}x+b \right |}{\left \| w \right \|}$
### 存在$\frac{\left | w^{T}x+b \right |}{\left \| w \right \|} >= d , (y = 1)$ 且 $\frac{\left | w^{T}x+b \right |}{\left \| w \right \|} <= -d , (y = -1)$
### 我们要求的便是max$\frac{\left | w^{T}x+b \right |}{\left \| w \right \|}$ => $max\frac{1}{\left \| w \right \|}$ , 即求min$\left \| w \right \|$ , 为了方便求导，使原函数为$\frac{1}{2}min\left \| w \right \|^2$
### 有一点是需要注意的，这个是有条件的最优化问题，条件为 $y^{(i)}(w^{T}x^{(i)} + b)\geq 1$
## Soft Margin SVM -- 适用于线性不可分问题
### 给条件添加一个宽松量$\varepsilon $
### 条件变成$y^{(i)}(w^{T}x^{(i)} + b)\geq 1 - \varepsilon _{i}$
### 给原函数添加正则项后为 ： $min\frac{1}{2}\left \| w \right \|^2 + C\sum_{i=1}^{m}\varepsilon _{i}$
# 解决非线性问题
### 使用多项式回归解决 （引入核函数）
## 核函数 ： 大大降低计算复杂度
### 多项式核函数 (以二项式为例) K(x,y) = (x·y + 1)$^2$
**K(x,y) = $\left ( \sum_{i=1}^{n}x_{i}y_{i} + 1 \right )^2$ = $\sum_{i=1}^{n}(x_{i}^2)(y_{i}^2)+\sum_{i=2}^{n}\sum_{j=1}^{i-1}(\sqrt{2}x_{i}x_{j})(\sqrt{2}y_{i}y_{j}) + \sum_{i=1}^{n}(\sqrt{2}x_{i})(\sqrt{2}y_{i}) + 1 = x^{'}y^{'}$**  
**其中$x^{'} = (x_{n}^{2} , ... , x_{1}^{2} , \sqrt{2}x_{n}x_{n-1} , ... , \sqrt{2}x_{n} , ... , \sqrt{2}x_{1} , 1)$**
### 多项式核函数的数学表达式为: $K(x,y) = (x·y + c)^d$
## RBF核函数（高斯核函数)
**思想 ：将每个样本点映射到一个无穷维的特征空间**
### K(x,y) = $e^{-\gamma \left \| x-y \right \|^2}$
### 高斯函数 ： g(x) = $\frac{1}{\sigma \sqrt{2}}e^{-\frac{1}{2}(\frac{x-u}{\sigma })^2}$
