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

### 2 参数更新
`计算参数的梯度后, 按公式更新参数`

$$
\mathbf w\leftarrow \mathbf w - \alpha \frac{\partial \cal R(\mathbf w,b)}{\partial \mathbf w}
$$

$$
\mathbf w\leftarrow \mathbf w - \alpha \frac{\partial \cal R(\mathbf w,b)}{\partial \mathbf w}
$$

其中 $\alpha$ 为学习率

> 将上面的参数更新过程包装为优化器,首先定义一个优化器基类Optimizer,方便后续所有的优化器调用。在这个基类中,需要初始化优化器的初始学习率init_lr,以及指定优化器需要优化的参数

### 3 评价指标
`在分类任务中, 通常使用准确率(Accuracy)作为评价指标`

$$
\mathcal{A} 
= 
\frac{1}{N}
	\sum_{n=1}^N
    I
    	(y^{(n)} = \hat{y}^{(n)})
$$

`其中` $I(·)$ 是指示函数

> 指示函数（Indicator Function）是一个 数学上的"开关函数", 它的值只有两种:
>
> 当条件成立时,取值为 1
>
> 当条件不成立时,取值为 0

**dim**: 表示沿着哪个维度进行操作

| 维度 | 含义（在2D张量里）| torch.argmax(..., dim=?) 的含义 |
|-----|-----|-----|
| dim=0 | 纵向（跨行） | 在每一“列”中找最大值（比较上下） |
| dim=1 | 横向（跨列） | 在每一“行”中找最大值（比较左右） |

# 多分类

## Softmax
`解决上下溢出`
Softmax函数可以将多个标量映射为一个概率分布,对于一个 $K$ 维向量, $\mathbf x=[x_1,\cdots,x_K]$

$$
\mathrm{softmax}(x_k) = \frac{\exp(x_k)}{\sum_{i=1}^K \exp(x_i)}
$$

在Softmax函数的计算过程中,要注意**上溢出**和**下溢出**的问题.假设Softmax 函数中所有的 $x_k$ 都是相同大小的数值 $a$, 理论上，所有的输出都应该为 $\frac{1}{k}$

- $a$ 为一个非常大的负数, 此时 $\exp(a)$ 会发生下溢出现象.计算机在进行数值计算时,当数值过小,会被四舍五入为0.此时，Softmax函数的分母会变为0，导致计算出现问题;
- $a$ 为一个非常大的正数, 此时会导致 $\exp(a)$ 发生上溢出现象，导致计算出现问题

为了解决上溢出和下溢出的问题, 在计算Softmax函数时, 可以使用 $x_k - \max(\mathbf x)$ 代替 $x_k$. 此时, 通过减去最大值, $x_k$ 最大为0, 避免了上溢出的问题; 同时，分母中至少会包含一个值为1的项, 从而也避免了下溢出的问题

## Softmax回归算子
在Softmax回归算子中, 类别标签 $y\in\{1,2,…,C\}$, 给定一个样本 $x$, 使用Softmax回归预测属于类别 $c$ 的条件概率为 

$$
\begin{aligned}
p(y=c|\mathbf x) &= \mathrm{softmax}(\mathbf w_c^T \mathbf x+b_c)
\end{aligned}
$$

其中 $\mathbf w_c$ 是类别 $c$ 的权重向量, $b_c$ 是类别 $c$ 的偏置项.

Softmax回归模型其实就是线性函数与Softmax函数的组合

将 $N$ 个样本归为一组进行成批地预测

$$
\hat{\mathbf Y} = \mathrm{softmax}(\boldsymbol{X} \boldsymbol{W} + \mathbf b)
$$

其中 $\boldsymbol{X}\in \mathbb{R}^{N\times D}$, 为 $N$ 个样本的特征矩阵, $\boldsymbol{W}=[\mathbf w_1,……,\mathbf w_C]$ 为 $C$ 个类的权重向量组成的矩阵, $\hat{\mathbf Y}\in \mathbb{R}^{C}$ 为所有类别的预测条件概率组成的矩阵

> 把 每个类别的权重向量作为 W 的一列, $\boldsymbol{w_c}\in \mathbb{R}^{D}$ $\boldsymbol{W}\in \mathbb{R}^{D \times C}$

> 每一个样本 $\boldsymbol{x}\in \mathbb{R}^{D}$, 将 $N$ 个样本归为一组进行成批地预测, $\boldsymbol{X}\in \mathbb{R}^{N \times D}$

> $\boldsymbol{XW}\in \mathbb{R}^{N \times C}$, 每个样本得到四个类别的得分

> 偏置项 $\boldsymbol{b}\in \mathbb{R}^{C}$

# b, W, A 变换？ 解释!

非常棒的问题 👏！
你抓住了核心本质：**偏置项（bias）确实可以让模型“平移”决策边界**。
我们可以从几种角度来理解它的作用👇

---

## 🧩 一、Softmax 回归的基本式子

对每个类别 ( c )，模型计算：
[
z_c = \mathbf{w}_c^\top \mathbf{x} + b_c
]
再用 Softmax：
[
p(y=c|\mathbf{x}) = \frac{e^{z_c}}{\sum_j e^{z_j}}
]

其中：

* ( \mathbf{w}_c^\top \mathbf{x} ) 是输入的 **线性组合（方向性）**
* ( b_c ) 是偏置项，用于 **平移整个函数的输出**

---

## 🧠 二、从几何角度看：平移决策边界

你可以把线性分类器看成一条超平面：
[
\mathbf{w}_c^\top \mathbf{x} + b_c = 0
]

这条平面将“属于类 c” 与 “不属于类 c” 的样本分开。

* ( \mathbf{w}_c )：决定了**方向（斜率）**
* ( b_c )：决定了**位置（截距 / 平移）**

👉 没有 (b_c) 时，所有超平面都会**穿过原点**。
这意味着分类边界会被迫绕着原点转，模型灵活性很低。

有了 (b_c)，分类边界就可以：

> “平移”到空间中任意位置，而不是固定在原点。

---

### 🔹 举个二维例子

假设：
[
z = w_1 x_1 + w_2 x_2 + b
]

决策边界是：
[
w_1 x_1 + w_2 x_2 + b = 0
]

* 当 (b = 0)：直线一定过原点
* 当 (b > 0)：整条线向下平移
* 当 (b < 0)：整条线向上平移

这样模型可以在平面上找到最合适的分割线。

---

## ⚙️ 三、与“线性变换 A” 的区别

你提到的数据生成代码：

```python
A = 2 * torch.rand((n_features, n_features)) - 1
X_k = torch.matmul(X_k, A)
```

这个 **A 是线性变换矩阵**，主要作用是：

* 旋转、拉伸、扭曲样本的分布；
* 改变数据的“方向”和“形状”。

而偏置 (b)：

* 不改变方向；
* 只负责整体**平移**，让模型能捕捉“原点不在中心”的分布。

---

## 🧩 四、直觉总结

| 元素     | 作用              | 类比    |
| ------ | --------------- | ----- |
| 权重 (W) | 控制方向、放缩特征影响力    | 旋转、拉伸 |
| 偏置 (b) | 平移输出空间、调整决策边界位置 | 平移    |

---

✅ **一句话总结：**
偏置项 (b) 就像是“线性变换后的坐标系原点的偏移量”，
没有它，所有决策都被困在“穿过原点”的世界里。

---

是否希望我画一个二维 Softmax/逻辑回归的决策边界图，直观展示「有/无偏置」的区别？
