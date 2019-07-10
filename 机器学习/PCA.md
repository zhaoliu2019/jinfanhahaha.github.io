# PCA（主成分分析）Principal Component Analysis
**1、一个非监督的机器学习算法  
2、主要用于数据的降维  
3、通过降维，可以发现更便于人类理解的特征  
4、其他应用，可视化，降噪**  
**第一步，将样例的均值归为0 （demean操作）  
由原来的Var(x) = $\frac{1}{m}\sum_{i=1}^{m}(x_{i}-\bar{x})^2$变成Var(x) = $\frac{1}{m}\sum_{i=1}^{m}x_{i}^2$**  
**第二步：找到一根轴w=(w1,w2,w3,···,wn)使得所有样本映射到w以后 , $Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{m}\left | X_{project}^{i} - \bar{X}_{project} \right |^2$ 最大**  
**因为经过demean操作，实际上使$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{m}\left | X_{project}^{i}\right |^2$最大**  
**1、$X^{(i)}w = \left \| X^{i} \right \|·\left \| w \right \|cos\theta $**  
**2、$X^{(i)}w = \left \| X^{i} \right \|cos\theta $**  
**3、$X^{(i)}w = \left \| X_{project}^{(i)} \right \|$**  
**最终我们使$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^{m}\left \| X^{(i)}w \right \|^2 = \frac{1}{m}\sum_{i=1}^{m}(\sum_{j=1}^{n}X_{j}^{(i)}w_{j})^2$ 最大**
