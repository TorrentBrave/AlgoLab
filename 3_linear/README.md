# 线性分类

`分类:`预测标签是一些离散的类别(符号), 根据分类任务的类别数量又可以分为二分类任务和多分类任务

`线性分类是指利用一个或多个线性函数将样本进行分类`

常用的线性分类模型:
- Logistic回归
- Softmax回归

<div>
    <img src="https://camo.githubusercontent.com/fcc3fe3fc86a956abf938c3bba67bd0e1cded8364035351539e94dbeb656eff1/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f37393763643238356662623134313132393434333738393731646466343032613761376633643866646334313436316562643433373362333737306133623436" alt="线性模型" width="800">
</div>

## 模型构建
`与线性回归一样,Logistic回归也会将输入特征与权重做线性叠加,不同之处在于，Logistic回归引入了非线性函数` $g:\mathbb{R}^D \rightarrow (0,1)$, 预测类别标签的后验概率 $p(y=1|\mathbf x)$, 从而解决连续的线性函数不适合进行分类的问题


`其中判别函数` $\sigma(\cdot)$ `为Logistic函数,也称为激活函数, 作用是将线性函数` $f(\mathbf x;\mathbf w,b)$ `的输出从实数区间“挤压”到（0,1）之间，用来表示概率.Logistic函数定义为`

## 损失函数
`模型训练过程中,需要使用损失函数来量化预测值和真实值之间的差异` $y$`表示样本` $x$ 的标签的真实概率分布, 向量 $\hat{\mathbf y}=p(\mathbf y|\mathbf x)$ 表示预测的标签概率分布, 通常使用 **交叉熵损失函数**, 给定 $y$ 的情况下, 如果预测的概率分布 $\hat{\mathbf y}$ 与标签真实的分布 $\mathbf y$, 则交叉熵越小, 如果 $p(\mathbf x)$ 和 $\mathbf y$ 越远, 交叉熵就越大.

对于二分类任务, 只需计算 $\hat{y}=p(y=1|\mathbf x)$, 用 $1-\hat{y}$ 表示 $p(y=0|\mathbf x)$, 给定N个训练样本的训练集 $\{(\mathbf x^{(n)},y^{(n)})\} ^N_{n=1}$ 使用 交叉熵损失函数,Logistic回归的风险函数计算方式为:

$\begin{aligned}
\cal R(\mathbf w,b) &= -\frac{1}{N}\sum_{n=1}^N \Big(y^{(n)}\log\hat{y}^{(n)} + (1-y^{(n)})\log(1-\hat{y}^{(n)})\Big)
\end{aligned}$

`向量形式可以表示为:`
$$
\begin{aligned}
\cal R(\mathbf w,b) &= -\frac{1}{N}\Big(\mathbf y^ \mathrm{ T }  \log\hat{\mathbf y} + (1-\mathbf y)^ \mathrm{ T } \log(1-\hat{\mathbf y})\Big)
\end{aligned}
$$

其中 $\mathbf y\in [0,1]^N$ 为 $N$ 个样本的 真实标签构成的 $N$ 维向量, $\hat{\mathbf y}$ 为 $N$ 个标本标签为1 的后验概率构成的 $N$ 维向量

## 模型优化

不同于线性回归中直接使用最小二乘法即可进行模型参数的求解，Logistic回归需要使用优化算法对模型参数进行有限次地迭代来获取更优的模型, 从而尽可能地降低风险函数的值。 在机器学习任务中，最简单、常用的优化算法是梯度下降法

梯度下降法进行模型优化, 需要初始化参数 $W$ 和 $b$, 然后不断计算它们的梯度, 并沿梯度的反方向更新参数

### 1 梯度计算
`Logistic回归中, 风险函数` $\cal R(\mathbf w,b)$ 关于参数 $w$ 和 $b$的偏导数:

$$
\begin{aligned}
\frac{\partial \cal R(\mathbf w,b)}{\partial \mathbf w} = -\frac{1}{N}\sum_{n=1}^N \mathbf x^{(n)}(y^{(n)}- \hat{y}^{(n)}) = -\frac{1}{N} \mathbf X^ \mathrm{ T }  (\mathbf y-\hat{\mathbf y})
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \cal R(\mathbf w,b)}{\partial b} = -\frac{1}{N}\sum_{n=1}^N (y^{(n)}- \hat{y}^{(n)}) = -\frac{1}{N} \mathbf {sum}(\mathbf y-\hat{\mathbf y})
\end{aligned}
$$

> 通常将偏导数的计算过程定义在Logistic回归算子的backward函数中