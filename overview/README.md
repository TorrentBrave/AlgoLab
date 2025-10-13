# 机器学习实践五要素
<div align="center">
  <img src="https://camo.githubusercontent.com/89a77d879340a7df32799b624a0dfd307719d33342999921ce926936b58a5f39/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f34613062613032636431383534323033623730303866383438326561613066353333613535653439393030633466623862343430363365663061346437353732" alt="机器学习系统" width="800">
</div>

`要通过机器学习来解决一个特定`任务时, 需要准备5方面要素:

1. 数据集: 收集任务相关的数据集用来模型训练和测试, 可分为训练集, 验证集和测试集;

2. 模型: 实现输入到输出的映射, 通常为可学习的函数

3. 学习准则: 模型优化的目标, 通常为损失函数和正则化项的加权组合

4. 优化算法: 根据学习准则优化机器学习模型的参数

5. 评价指标: 用来评价学习到的机器学习模型的性能

<div align="center">
  <img src="https://camo.githubusercontent.com/c036198c25e7a3ccd33fdd42e116d173cd73c86a9a4bcbfa6c30256c14aeeeb4/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f63363635393934396365336234633436383361336638326437623936383362306433343165653537623837663466623038346265353865333164626537663932" alt="机器学习系统" width="800">
</div>

## 1.1 数据
`噪音越少、规模越大、覆盖范围越广的数据集往往能训练出性能更好的模型`

***数据预处理：***

> 对收集到的数据进行基本预处理, 统计、特征归一化和异常值处理等

> 再将数据划分为训练集、验证集(开发集)、测试集
- **训练集**: 用于模型训练时调整模型的参数，在这份数据集上的误差被称为训练误差
- **验证集(开发集)**: 对于复杂的模型，常常有一些超参数需要调节，因此需要尝试多种超参数的组合来分别训练多个模型，然后对比它们在验证集上的表现，选择一组相对最好的超参数，最后才使用这组参数下训练的模型在测试集上评估测试误差
- **测试集**: 模型在这份数据集上的误差被称为测试误差. 训练模型的目的是为了通过从训练数据中找到规律来预测未知数据, 因此测试误差更能反映出模型表现的指标

数据划分时要考虑到两个因素：
> 更多的训练数据会降低参数估计的方差,从而得到更可信的模型;而更多的测试数据会降低测试误差的方差,从而得到更可信的测试误差.如果给定的数据集没有做任何划分,我们一般可以大致按照7:3或者8:2的比例划分训练集和测试集,再根据7:3或者8:2的比例从训练集中再次划分出训练集和验证集

---

> 需要强调的是，测试集只能用来评测模型最终的性能，在整个模型训练过程中不能有测试集的参与

## 1.2 模型
`在数据上训练模型, 希望计算机从一个函数集合` $\mathcal{F} = \{f_1(\boldsymbol{x}), f_2(\boldsymbol{x}), \cdots \}$ `中自动寻找一个"最优"的函数`$f^∗(\boldsymbol{x})$ `来近似每个样本的特征向量` $x$ `和标签` $y$ `之间的真实映射关系, 这个函数集合也称为` **假设空间** , `实际问题中, 假设空间` $\mathcal{F}$ `通常为一个参数化的函数族`

$$
\mathcal{F}=\left\{f(\boldsymbol{x} ; \theta) \mid \theta \in \mathbb{R}^{D}\right\}
$$

$f(\boldsymbol{x} ; \theta)$ `是参数为` $\theta $ `的函数`, `也称为模型`, $D$ `为参数的数量`

`常见的假设空间可分为线性和非线性两种`,`对应的模型` $f$ `也分别称为线性模型和非线性模型`.**线性模型**`的假设空间为一个参数化的线性函数族`:

$$
f(\boldsymbol{x} ; \theta)=\boldsymbol{w}^{\top} \boldsymbol{x}+b
$$

`参数`$\theta $`包含了权重向量`$w$ `和偏置` $b$

`线性模型可以由` **非线性基函数** $\phi(\boldsymbol{x})$ `变为` **非线性函数**, `从而增强模型能力`:

$$
f(\boldsymbol{x} ; \theta)=\boldsymbol{w}^{\top} \phi(\boldsymbol{x})+b
$$

其中 $\phi(\boldsymbol{x})=\left[\phi_{1}(\boldsymbol{x}), \phi_{2}(\boldsymbol{x}), \cdots, \phi_{K}(\boldsymbol{x})\right]^{\top}$ 为 $K$ 个非线性基函数组成的向量


$\phi(\boldsymbol{x})=\left[\phi_{1}(\boldsymbol{x}),
\phi_{2}(\boldsymbol{x}), \cdots,
\phi_{K}(\boldsymbol{x})\right]^{\top}$ 中的每个
$\phi_k(\boldsymbol{x})$ 是对整个输入向量 $\boldsymbol{x}$
进行变换，而不是分别对每一维变换。

***具体说明：***

1. 输入：$\boldsymbol{x} \in \mathbb{R}^n$（n维向量）
2. 每个基函数：$\phi_k: \mathbb{R}^n \rightarrow
\mathbb{R}$（接受整个向量，输出标量）
3. 输出：$\phi(\boldsymbol{x}) \in
\mathbb{R}^K$（K维特征向量）



输入 $\boldsymbol{x} = [x_1, x_2]$

隐藏层有3个神经元：

• 第1个神经元输出：$\phi_1(\boldsymbol{x}) =
\text{ReLU}(2x_1 + 3x_2 + 1)$
• 第2个神经元输出：$\phi_2(\boldsymbol{x}) = \text{ReLU}(-
x_1 + x_2 - 2)$
• 第3个神经元输出：$\phi_3(\boldsymbol{x}) =
\text{ReLU}(x_1 - x_2 + 0.5)$

把这些输出组成向量： $$\phi(\boldsymbol{x}) =
\begin{bmatrix} \phi_1(\boldsymbol{x}), \
\phi_2(\boldsymbol{x}), \ \phi_3(\boldsymbol{x})
\end{bmatrix}$$

这就是 $\phi(\boldsymbol{x})$ 的含义

---
***注意:***
> $\boldsymbol{x} \in \mathbb{R}^D$ $D$ 维列向量
> 
> $[x_1, \cdots, x_D]$ $D$ 维行向量
>
> $[x_1, \cdots, x_D]^\top$ or $[x_1; \cdots; x_D]$ $D$ 维列向量

---

## 1.3 学习准则
`为了衡量一个模型的好坏,需要定义一个损失函数` $\mathcal{L}(\boldsymbol{y},f(\boldsymbol{x};\theta))$. `损失函数是一个非负实数函数,用来量化模型预测标签和真实标签之间的差异. 常见的损失函数有 0-1损失, 平方损失函数, 交叉熵损失函数等`.

***机器学习的目标***：

`找到最优的模型` $𝑓(𝒙;\theta^∗)$ `在真实数据分布上损失函数的期望最小. 然而在实际中, 我们无法获得真实数据分布, 通常会用在训练集上的平均损失替代.`

`一个模型在训练集` $\mathcal{D}=\{(\boldsymbol{x}^{(n)},y^{(n)})\}_{n=1}^N$ `上的平均损失称为` **经验风险(Empirical Risk)**：

$$
\mathcal{R}^{emp}_\mathcal{D}(\theta)=\frac{1}{N}\sum_{n=1}^{N}\mathcal{L}(y^{(n)},f({x}^{(n)};\theta))
$$

> 通常情况下,我们可以通过使得经验风险最小化来获得具有预测能力的模型.然而,当模型比较复杂或训练数据量比较少时,经验风险最小化获得的模型在测试集上的效果比较差.而模型在测试集上的性能才是我们真正关心的指标.当一个模型在训练集错误率很低,而在测试集上错误率较高时,通常意味着发生了过拟合（Overfitting）现象.为了缓解模型的过拟合问题,我们通常会在经验损失上加上一定的正则化项来限制模型能力

> 过拟合通常是由于模型复杂度比较高引起的。在实践中，最常用的正则化方式有对模型的参数进行约束，比如 $\ell_1$ 或者 $\ell_2$ 范数约束.这样,我们就得到了结构风险（Structure Risk）

$$
\mathcal{R}^{struct}_{\mathcal{D}}(\theta)=\mathcal{R}^{emp}_{\mathcal{D}}(\theta)+\lambda \ell_p(\theta)
$$

> 其中 $\lambda$ 为正则化系数, $p = 1$ 或 $2$ 表示 $\ell_1$ 或者 $\ell_2$ 范数

---
***模型复杂度的解释***

`模型复杂度 ≠ 模型结构数量,而是"拟合能力的灵活性"`

`"明明都是一个模型,只不过里面的数值变了",这没错——模型的结构（比如神经网络层数、多项式阶数）没变.但复杂度不仅由结构决定,还由参数的取值范围决定`

> 模型复杂度指的是:模型能够拟合多么"奇怪"或"剧烈波动"的函数能力

`从泛化误差界理解：复杂度与参数范数相关`

`在理论中，泛化误差（测试误差）可以被上界表示为` 
 
$$Test Error ≤ Training Error + Complexity Penalty$$

而这个 Complexity Penalty 往往与参数的范数有关,例如： 

    对于线性分类器，VC维或Rademacher复杂度与 ∥θ∥2​  成正比；
    对于神经网络，泛化界常包含权重范数的乘积。
     
所以,减小 $∥θ∥p$​ 直接降低了理论上的泛化误差上界,说明模型“更简单”、更不容易过拟合

---
***正则化作用:***

正则化通过限制模型参数的大小（如使用 ℓ₁ 或 ℓ₂ 范数），并非改变模型结构，而是降低其“有效复杂度”。原因包括：

1. **大参数使模型对输入过于敏感，容易拟合噪声；小参数让函数更平滑、稳定。**  
2. **正则化缩小了模型可选的函数空间（假设空间），只保留更简单的函数。**  
3. **在高阶模型（如多项式）中，小参数会自动抑制高阶项，等效于简化模型。**  
4. **理论泛化误差界与参数范数相关，小范数意味着更好的泛化能力。**

因此，正则化是在**拟合能力与模型简洁性之间取得平衡**，从而缓解过拟合，这正是结构风险最小化的核心思想。

[Qwen_Answer](https://chat.qwen.ai/c/b8a4c4a9-78f2-46c0-9acb-2494520021ba)

---

## 1.4 优化算法
`有了优化目标后,机器学习问题就转化为优化问题,我们可以利用已知的优化算法来学习最优的参数.`

`当优化函数为凸函数时,可以令参数的偏导数等于0来计算最优参数的解析解.当优化函数为非凸函数时,我们可以用一阶的优化算法来进行优化`

- 若优化目标（损失函数）是**凸函数**（如线性回归的MSE）,其全局最小值可通过**令梯度为零直接求解析解**  
- 若为**非凸函数**（如神经网络损失）,无法求解析解，且存在多个局部极小值, 需使用**一阶迭代优化算法**（如梯度下降、Adam）逐步逼近较优解
- 核心区别：**凸问题有唯一最优解且可直接计算；非凸问题只能数值逼近，不保证全局最优**, 一阶优化算法只用到一阶导数, 即梯度

## 1.5 评估指标
`(Metric) 用于评价模型效果，即给定一个测试集，用模型对测试集中的每个样本进行预测，并根据预测结果计算评价分数。回归任务的评估指标一般有预测值与真实值的均方差，分类任务的评估指标一般有准确率、召回率、F1值等`

`对于一个机器学习任务，一般会先确定任务类型，再确定任务的评价指标，再根据评价指标来建立模型，选择学习准则。由于评价指标不可微等问题有时候学习准则并不能完全和评价指标一致，我们往往会选择一定的损失函数使得两者尽可能一致`

# 实现线性回归模型
`回归任务是一类典型的监督机器学习任务，对自变量和因变量之间关系进行建模分析，其预测值通常为一个连续值，比如房屋价格预测、电源票房预测等。线性回归(Linear Regression)是指一类利用线性函数来对自变量和因变量之间关系进行建模的回归任务，是机器学习和统计学中最基础和最广泛应用的模型`

`使用最小二乘法来求解参数`

## 2.1 线性最小二乘法求解析解

这是一个非常关键且经典的问题！我们来一步步推导 **线性最小二乘的解析解**，也就是为什么最优参数是：

$$
\boldsymbol{\theta}^* = (X^\top X)^{-1} X^\top \mathbf{y}
$$

---

### 问题设定

我们有：
- 输入数据矩阵 $X \in \mathbb{R}^{n \times d}$：每行是一个样本（共 $n$ 个样本，$d$ 个特征）
- 真实标签向量 $\mathbf{y} \in \mathbb{R}^n$
- 线性模型：$\hat{\mathbf{y}} = X \boldsymbol{\theta}$，其中 $\boldsymbol{\theta} \in \mathbb{R}^d$ 是待求参数

目标是最小化 **平方误差和（Sum of Squared Errors）**：

$$
\mathcal{L}(\boldsymbol{\theta}) = \|\mathbf{y} - X\boldsymbol{\theta}\|_2^2 = (\mathbf{y} - X\boldsymbol{\theta})^\top (\mathbf{y} - X\boldsymbol{\theta})
$$

这是一个关于 $\boldsymbol{\theta}$ 的**凸二次函数**，存在全局最小值。

---

### 🔍 推导步骤

#### 第一步：展开损失函数

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta}) 
&= (\mathbf{y} - X\boldsymbol{\theta})^\top (\mathbf{y} - X\boldsymbol{\theta}) \\
&= \mathbf{y}^\top \mathbf{y} - \mathbf{y}^\top X \boldsymbol{\theta} - \boldsymbol{\theta}^\top X^\top \mathbf{y} + \boldsymbol{\theta}^\top X^\top X \boldsymbol{\theta}
\end{aligned}
$$

注意：$\mathbf{y}^\top X \boldsymbol{\theta}$ 是一个标量，等于它的转置，即：
$$
\mathbf{y}^\top X \boldsymbol{\theta} = (\mathbf{y}^\top X \boldsymbol{\theta})^\top = \boldsymbol{\theta}^\top X^\top \mathbf{y}
$$

所以两项可以合并：

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbf{y}^\top \mathbf{y} - 2 \boldsymbol{\theta}^\top X^\top \mathbf{y} + \boldsymbol{\theta}^\top X^\top X \boldsymbol{\theta}
$$

---

#### 第二步：对 $\boldsymbol{\theta}$ 求梯度（偏导）

使用矩阵微分规则（记住：$\frac{\partial}{\partial \boldsymbol{\theta}} (\boldsymbol{\theta}^\top A \boldsymbol{\theta}) = (A + A^\top)\boldsymbol{\theta}$，若 $A$ 对称则为 $2A\boldsymbol{\theta}$）：

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) 
= -2 X^\top \mathbf{y} + 2 X^\top X \boldsymbol{\theta}
$$

---

#### 第三步：令梯度为零（极小值条件）

$$
-2 X^\top \mathbf{y} + 2 X^\top X \boldsymbol{\theta} = 0
$$

两边除以 2：

$$
X^\top X \boldsymbol{\theta} = X^\top \mathbf{y}
$$

这个方程叫做 **正规方程（Normal Equation）**。

---

#### 第四步：解出 $\boldsymbol{\theta}$

如果 $X^\top X$ 是**可逆的**（即满秩，通常要求 $n \geq d$ 且特征线性无关），则两边左乘 $(X^\top X)^{-1}$：

$$
\boldsymbol{\theta}^* = (X^\top X)^{-1} X^\top \mathbf{y}
$$

✅ 这就是线性最小二乘的**解析解**！

---

### 💡 直观理解

- $X^\top \mathbf{y}$：表示输入特征与目标值的**相关性**（协方差类信息）
- $X^\top X$：表示特征之间的**自相关矩阵**（也叫 Gram 矩阵）
- $(X^\top X)^{-1} X^\top$：整体是一个**投影算子**，把 $\mathbf{y}$ 投影到 $X$ 的列空间中，得到最佳线性逼近

换句话说：**最小二乘解就是把真实标签 $\mathbf{y}$ 投影到由特征张成的线性子空间上**，使得误差向量 $\mathbf{y} - X\boldsymbol{\theta}$ 与该子空间正交（即残差与所有特征无关）。

---

### ⚠️ 注意条件

- 解存在的前提是 $X^\top X$ **可逆**（即 $X$ 列满秩）；
- 如果不可逆（如特征过多、存在共线性），需用：
  - 岭回归（加 $\ell_2$ 正则化）：$(X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$
  - 伪逆（Moore-Penrose inverse）：$\boldsymbol{\theta}^* = X^+ \mathbf{y}$

---

### ✅ 总结

> 最小二乘的解析解 $\boldsymbol{\theta}^* = (X^\top X)^{-1} X^\top \mathbf{y}$  
> 是通过对平方损失函数求导、令梯度为零、解正规方程得到的。  
> 它的本质是：**在特征张成的线性空间中，找到最接近真实标签的投影点**

## 2.2 数据集构造
`构造一个小的回归数据集, 假设输入特征和输出标签的维度都为1, 需要被拟合的函数定义为:`

```python
# 真实函数的参数缺省值为 
```