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

这是一个非常关键且经典的问题！我们来一步步推导 **线性最小二乘的解析解**, 也就是为什么最优参数是：

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

你的理解基本正确！我们来清晰梳理：

---

### ✅ 1. `torch.rand` 和 `torch.normal` 的分布

| 函数 | 分布类型 | 参数 | 说明 |
|------|--------|------|------|
| `torch.rand(shape)` | **均匀分布（Uniform）** | 范围默认是 $[0, 1)$ | 从 0 到 1 均匀随机采样 |
| `torch.normal(mean, std)` | **正态分布（高斯分布，Gaussian/Normal）** | `mean`（均值）、`std`（标准差） | 以 `mean` 为中心、`std` 为离散程度的钟形分布 |

> 📌 补充：`torch.randn(shape)` 是 `torch.normal(0, 1)` 的简写（标准正态分布）。

---

### ✅ 2. 为什么用不同分布？——背后的建模假设

#### （1）**用均匀分布生成输入数据（如 `X`）**
- **目的**：让输入覆盖整个感兴趣的区间，**无偏好地采样**。
- **例子**：你想拟合一个函数在 $[-5, 5]$ 上的行为，就用 `torch.rand` 生成 $[0,1)$，再线性变换到 $[-5,5]$。
- **为什么不用高斯？**  
  高斯分布集中在均值附近，两端样本极少，可能导致模型在边界区域欠拟合。**均匀采样能更公平地探索整个输入空间**。

✅ 所以：**均匀分布用于“主动设计实验”**，确保数据覆盖全面。

---

#### （2）**用高斯分布模拟标签噪声（如 `y = f(x) + ε`, ε ~ N(0, σ²)）**
- **原因 1：中心极限定理**  
  现实中的测量误差、环境干扰等，往往是**大量微小独立随机因素叠加**的结果，根据中心极限定理，其总和近似服从高斯分布。
  
- **原因 2：数学性质优良**  
  - 高斯分布由均值和方差完全刻画，分析方便；
  - 最大似然估计下，**高斯噪声 ⇨ 最小二乘损失（MSE）**，这是回归问题的基石；
  - 在贝叶斯推断、卡尔曼滤波等理论中，高斯假设带来解析解。

- **原因 3：符合现实观测**  
  大量实验表明，传感器噪声、人为标注误差等**确实近似服从高斯分布**（或近似对称、单峰、轻尾）。

✅ 所以：**高斯噪声是对现实误差最合理、最简洁的建模假设**。

---

### 举个例子

假设真实函数是 $y = 2x + 1$，你生成训练数据：

```python
x = torch.rand(100) * 20 - 10        # 均匀采样 x ∈ [-10, 10]
noise = torch.normal(0, 0.5, size=(100,))  # 高斯噪声
y = 2 * x + 1 + noise                # 带噪声的标签
```

- `x` 均匀分布 → 模型能在整个区间学习；
- `noise` 高斯分布 → 模拟真实世界中围绕真实值上下波动的随机误差。

---

### ❓ 那能不能反过来？

- **用高斯分布生成 `x`？**  
  可以，但只适用于你**关心中心区域**的场景（比如人脸识别中人脸位置集中在图像中央）。

- **用均匀噪声？**  
  也可以（比如 `y += torch.rand(...) - 0.5`），但：
  - 均匀噪声有硬边界（比如 ±0.5），现实中误差通常没有严格界限；
  - 均匀噪声的“极端值”概率和中间值一样高，不符合大多数物理/人为误差的特性。

> 🔬 **高斯噪声是默认选择，除非有特殊理由用其他分布**。

---

### ✅ 总结

| 用途 | 推荐分布 | 原因 |
|------|--------|------|
| **生成输入特征 `x`** | 均匀分布 (`torch.rand`) | 全面覆盖输入空间，无偏采样 |
| **模拟标签噪声 `ε`** | 高斯分布 (`torch.normal`) | 符合现实误差特性，理论优美，与 MSE 损失天然匹配 |

这不仅是习惯，更是**基于统计学原理和实践经验的合理建模选择**

```python
# 真实函数的参数缺省值为 
```

## 2.3 模型构建
`线性回归中, 自变量为样本的特征向量` $\boldsymbol{x}\in \mathbb{R}^D$ `每一维对应一个自变量`, `因变量是连续值的标签` $y\in R$, 线性模型定义为:
$$
f(\boldsymbol{x};\boldsymbol{w},b)=\boldsymbol{w}^T\boldsymbol{x}+b
$$
`其中权重向量` $\boldsymbol{w}\in \mathbb{R}^D$ `和偏置` $b\in \mathbb{R}$ `都是可学习的参数`
- **增广权重向量**定义模型能保持表达的简洁性
- **非增广权重向量**能保持和代码的一致性

实践中, 为了提高预测样本的效率, 通常会将 $N$ 样本归为一组进行成批地预测, 可以更好地利用GPU设备的并行能力
$$
\boldsymbol{y} =\boldsymbol{X} \boldsymbol{w} + b
$$
`其中` $\boldsymbol{X}\in \mathbb{R}^{N\times D}$ 为 $N$ 个样本的特征矩阵, $\boldsymbol{y}\in \mathbb{R}^N$ 为 $N$ 个预测值组成的列向量. ***它会是被实现的线性算子***

`实践中, 样本的矩阵` $X$ `是由` $N$ `个` $x$的 `行向量组成`, 教材中 $x$ `为列向量`, `特征矩阵和教材中的特征向量刚好是转置关系`

### 线性算子
- $X: tensor, shape=[N,D]$
- $y_{pred}: tensor, shape=[N]$
- $w: shape=[D,1]$
- $b: shape=[1]$
- $y_{pred} = torch.matmul(X,w)+b$

## 2.4 损失函数
`回归任务是对`**连续值**的预测, 希望模型能根据数据的特征输出一个连续值作为预测值, 因此回归任务中常用的评估指标 是 **均方误差**
令 $\boldsymbol{y}\in \mathbb{R}^N$, $\hat{\boldsymbol{y}}\in \mathbb{R}^N$ 分别为 $N$ 个样本的真实标签和预测标签, 均方误差的定义:

$$
\mathcal{L}(\boldsymbol{y},\hat{\boldsymbol{y}})=\frac{1}{2N}\|\boldsymbol{y}-\hat{\boldsymbol{y}}\|^2=\frac{1}{2N}\|\boldsymbol{X}\boldsymbol{w}+\boldsymbol{b}-\boldsymbol{y}\|^2
$$

`其中` $b$ 为 $N$ 维向量, 所有元素取值都为 $b$

## 2.5 模型优化
**经验风险最小化**, 线性回归可以通过最小二乘法求出参数, $w$ 和 $b$ 的解析解

$$\frac{\partial \mathcal{L}(\boldsymbol{y},\hat{\boldsymbol{y}})}{\partial b} = \mathbf{1}^T (\boldsymbol{X}\boldsymbol{w}+\boldsymbol{b}-\boldsymbol{y})$$

`其中`$1$为$N$维的全1向量, 为了简洁省略了均方误差的系数, $\frac{1}{N}$, **并不影响最后的结果**, `令上式等于0, 得到:`

$$
b^* =\bar{y}-\bar{\boldsymbol{x}}^T \boldsymbol{w}
$$

`其中`$\bar{y} = \frac{1}{N}\mathbf{1}^T\boldsymbol{y}$为所有标签的平均值, $\bar{\boldsymbol{x}} = \frac{1}{N}(\mathbf{1}^T \boldsymbol{X})^T$, 为所有特征向量的平均值, 将 $b^*$ 代入均方误差对参数 $w$ 的偏导数, 得到:

$$
\frac{\partial \mathcal{L}(\boldsymbol{y},\hat{\boldsymbol{y}})}{\partial \boldsymbol{w}} = (\boldsymbol{X}-\bar{\boldsymbol{x}}^T)^T \Big((\boldsymbol{X}-\bar{\boldsymbol{x}}^T)\boldsymbol{w}-(\boldsymbol{y}-\bar{y})\Big)
$$

令上式等于 0, 得到最优参数:

$$
\boldsymbol{w}^*=\Big((\boldsymbol{X}-\bar{\boldsymbol{x}}^T)^T(\boldsymbol{X}-\bar{\boldsymbol{x}}^T)\Big)^{\mathrm{-}1}(\boldsymbol{X}-\bar{\boldsymbol{x}}^T)^T (\boldsymbol{y}-\bar{y})
$$

$$
b^* =  \bar{y}-\bar{\boldsymbol{x}}^T \boldsymbol{w}^*
$$

若对参数$w$加上$l_2$正则化系数, $\boldsymbol{I}\in \mathbb{R}^{D\times D}$ 为单位矩阵,$\lambda>0$为预先设置的正则化系数

$$
\boldsymbol{w}^*=\Big((\boldsymbol{X}-\bar{\boldsymbol{x}}^T)^T(\boldsymbol{X}-\bar{\boldsymbol{x}}^T)+\lambda \boldsymbol{I}\Big)^{\mathrm{-}1}(\boldsymbol{X}-\bar{\boldsymbol{x}}^T)^T (\boldsymbol{y}-\bar{y})
$$

参数学习的过程通过优化器完成。由于我们可以基于最小二乘方法可以直接得到线性回归的解析解，此处的训练是求解析解的过程

## 什么时候进行转置

这是一个非常好的问题，涉及到**矩阵维度一致性**和**向量表示习惯**（行向量 vs 列向量）。

---

### 🔍 背景：数据矩阵 $\boldsymbol{X}$ 的形状

在你给出的上下文中，线性回归的输入数据矩阵 $\boldsymbol{X} \in \mathbb{R}^{N \times D}$，其中：
- $N$：样本数量
- $D$：特征维度
- **每一行是一个样本**（这是机器学习中最常见的约定）

例如：
$$
\boldsymbol{X} = 
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1D} \\
x_{21} & x_{22} & \cdots & x_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{ND}
\end{bmatrix}
$$

---

### 📌 目标：计算特征的均值向量 $\bar{\boldsymbol{x}}$

我们希望得到一个 **$D$ 维的列向量**，表示每个特征在所有样本上的平均值：

$$
\bar{\boldsymbol{x}} = 
\begin{bmatrix}
\frac{1}{N} \sum_{i=1}^N x_{i1} \\
\frac{1}{N} \sum_{i=1}^N x_{i2} \\
\vdots \\
\frac{1}{N} \sum_{i=1}^N x_{iD}
\end{bmatrix}
\in \mathbb{R}^{D}
$$

---

### 为什么用 $\frac{1}{N}(\mathbf{1}^T \boldsymbol{X})^T$？

1. **$\mathbf{1} \in \mathbb{R}^N$** 是全 1 列向量。
2. **$\mathbf{1}^T \boldsymbol{X}$** 的维度是：
   $$
   (1 \times N) \cdot (N \times D) = (1 \times D)
   $$
   结果是一个 **行向量**，其第 $j$ 个元素是 $\sum_{i=1}^N x_{ij}$，即第 $j$ 个特征的总和。

3. 但我们要的是 **列向量**（$D \times 1$），所以需要 **转置**：
   $$
   (\mathbf{1}^T \boldsymbol{X})^T \in \mathbb{R}^{D \times 1}
   $$

4. 再除以 $N$，就得到均值列向量：
   $$
   \bar{\boldsymbol{x}} = \frac{1}{N} (\mathbf{1}^T \boldsymbol{X})^T
   $$

---

### ✅ 举个数值例子

设：
$$
\boldsymbol{X} = 
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}, \quad
\mathbf{1} = 
\begin{bmatrix}
1 \\ 1 \\ 1
\end{bmatrix}
$$

则：
$$
\mathbf{1}^T \boldsymbol{X} = 
\begin{bmatrix}
1 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
= 
\begin{bmatrix}
9 & 12
\end{bmatrix} \quad \text{（行向量）}
$$

转置后：
$$
(\mathbf{1}^T \boldsymbol{X})^T = 
\begin{bmatrix}
9 \\ 12
\end{bmatrix}
\quad \Rightarrow \quad
\bar{\boldsymbol{x}} = \frac{1}{3}
\begin{bmatrix}
9 \\ 12
\end{bmatrix}
=
\begin{bmatrix}
3 \\ 4
\end{bmatrix}
$$

✅ 这正是每列的平均值，且是**列向量**。

---

### ❓ 什么时候需要转置？

| 情况 | 是否需要转置 | 原因 |
|------|------------|------|
| **$\boldsymbol{X}$ 每行是一个样本**（$N \times D$） | ✅ 需要 | $\mathbf{1}^T \boldsymbol{X}$ 是行向量，但均值通常表示为列向量 |
| **$\boldsymbol{X}$ 每列是一个样本**（$D \times N$） | ❌ 不需要 | $\boldsymbol{X} \mathbf{1}$ 直接得到列向量 |

> 📌 在绝大多数机器学习库（如 scikit-learn、PyTorch、TensorFlow）中，**数据矩阵都是“样本在行”**（$N \times D$），所以**需要转置**来得到列向量形式的均值。

---

### 💡 补充：也可以写成 $\bar{\boldsymbol{x}} = \frac{1}{N} \boldsymbol{X}^T \mathbf{1}$

注意：
$$
(\mathbf{1}^T \boldsymbol{X})^T = \boldsymbol{X}^T \mathbf{1}
$$

所以更常见的写法其实是：
$$
\bar{\boldsymbol{x}} = \frac{1}{N} \boldsymbol{X}^T \mathbf{1}
$$

这直接得到列向量，**无需显式写转置**，也更符合线性代数习惯。

---

### ✅ 总结

> **之所以写成 $\bar{\boldsymbol{x}} = \frac{1}{N}(\mathbf{1}^T \boldsymbol{X})^T$，是因为：**
>
> 1. $\mathbf{1}^T \boldsymbol{X}$ 计算的是每列（每个特征）的总和，结果是 **行向量**；
> 2. 但我们需要均值是 **列向量**（与权重 $\boldsymbol{w}$ 维度一致，便于计算 $\bar{\boldsymbol{x}}^T \boldsymbol{w}$）；
> 3. 因此必须 **转置**，使其从 $(1 \times D)$ 变为 $(D \times 1)$。
>
> **只要数据矩阵是“样本在行”**（$N \times D$），计算列均值时就需要这个转置（或等价地用 $\boldsymbol{X}^T \mathbf{1}$）

## 为什么进行正则化

你问了一个非常核心的问题：**为什么要在最优参数 $\boldsymbol{w}^*$ 上加 $ \ell_2 $ 正则化**（即岭回归）？

我们从**动机、问题、解决方案**三个层面来解释。

---

### 🔍 1. **不加正则化时，可能遇到什么问题**？

你给出的原始解是：
$$
\boldsymbol{w}^* = \left( (\boldsymbol{X} - \bar{\boldsymbol{x}}^T)^T (\boldsymbol{X} - \bar{\boldsymbol{x}}^T) \right)^{-1} (\boldsymbol{X} - \bar{\boldsymbol{x}}^T)^T (\boldsymbol{y} - \bar{y})
$$

这个解成立的前提是：  
> **矩阵 $ \boldsymbol{X}^T \boldsymbol{X} $（或中心化后的）必须可逆**（即满秩）。

但在实际中，常常出现以下情况：

#### ❌ 问题 1：**特征数 $D$ 大于样本数 $N$**（高维小样本）
- 例如：基因数据（$D=20000$ 个基因，$N=100$ 个病人）
- 此时 $\boldsymbol{X} \in \mathbb{R}^{N \times D}$，秩最多为 $N < D$
- 所以 $\boldsymbol{X}^T \boldsymbol{X}$ 是 **奇异矩阵**（不可逆），**解析解不存在**！

#### ❌ 问题 2：**特征之间高度相关**（共线性）
- 例如：`身高（cm）` 和 `身高（inch）` 同时作为特征
- 导致 $\boldsymbol{X}^T \boldsymbol{X}$ **接近奇异**，数值不稳定
- 微小的数据扰动会导致 $\boldsymbol{w}^*$ 剧烈变化（过拟合）

#### ❌ 问题 3：**过拟合**
- 即使矩阵可逆，如果模型太复杂（比如高维），$\boldsymbol{w}^*$ 可能取**非常大的值**去拟合训练噪声
- 虽然训练误差低，但测试误差高

---

### ✅ 2. **加 $\ell_2$ 正则化如何解决这些问题**？

正则化后的目标函数变为：
$$
\mathcal{L}_{\text{reg}} = \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{w} - b\|^2 + \lambda \|\boldsymbol{w}\|^2
$$

对应的解是：
$$
\boldsymbol{w}^* = \left( \boldsymbol{X}_c^T \boldsymbol{X}_c + \lambda \boldsymbol{I} \right)^{-1} \boldsymbol{X}_c^T \boldsymbol{y}_c
$$
（其中 $\boldsymbol{X}_c, \boldsymbol{y}_c$ 是中心化后的数据）

#### ✅ 效果 1：**保证矩阵可逆**
- $\boldsymbol{X}_c^T \boldsymbol{X}_c$ 是半正定的，加上 $\lambda \boldsymbol{I}$（$\lambda > 0$）后变成**正定矩阵**
- **一定可逆**！即使 $D > N$ 或存在共线性

#### ✅ 效果 2：**抑制参数过大**
- 正则项 $\lambda \|\boldsymbol{w}\|^2$ 惩罚大权重
- 优化器会在“拟合数据”和“保持 $\boldsymbol{w}$ 小”之间权衡
- 得到的 $\boldsymbol{w}^*$ 更平滑、更稳定

#### ✅ 效果 3：**缓解过拟合，提升泛化能力**
- 小的 $\boldsymbol{w}$ 意味着模型对输入不敏感，不容易拟合噪声
- 测试误差通常更低（尤其在小样本或高维场景）

---

### 直观例子：拟合一条直线

假设你只有 2 个点，但用 10 次多项式去拟合：
- 无正则化：曲线剧烈震荡，完美过拟合两点
- 加 $\ell_2$ 正则化：曲线更平滑，虽然训练误差略高，但更接近真实趋势

> **正则化 = 用一点偏差（bias）换大幅降低方差（variance）**

---

### 📌 为什么是 $\ell_2$ 而不是 $\ell_1$？
- $\ell_2$（岭回归）：**保持所有特征，但缩小权重**，解析解存在，计算高效
- $\ell_1$（Lasso）：**会把某些权重压到 0，实现特征选择**，但无解析解，需迭代优化

在线性回归中，**$\ell_2$ 是最常用、最稳定的正则化方式**。

---

### ✅ 总结：为什么要加正则化？

| 问题 | 正则化的作用 |
|------|------------|
| 矩阵不可逆（$D > N$） | ✅ 使 $\boldsymbol{X}^T\boldsymbol{X} + \lambda I$ 可逆 |
| 特征共线性 | ✅ 提高数值稳定性 |
| 过拟合 | ✅ 限制参数大小，提升泛化能力 |
| 参数过大 | ✅ 引入先验：更相信“小权重”模型 |

> **加 $\ell_2$ 正则化不是为了“改变最优解”，而是为了在现实约束下（数据少、特征多、噪声大）得到一个“更可靠、更泛化”的解。**

这就是**结构风险最小化**（Structural Risk Minimization）的思想：  
**不仅拟合训练数据，还要控制模型复杂度**

## 2.6 模型训练

`在准备了数据, 模型, 损失函数和参数学习的实现后, 开始模型的训练, 回归任务中, 模型的评价指标和损失函数一致, 都是均方误差`

`通过之前实现的线性回归来拟合训练数据, 并输出模型在训练集上的损失`

```python
input_size = 1
model 

```

`squeeze()压缩所有size=1的维度`
```python
x = torch.randn(3, 1, 4, 1)  # shape: (3, 1, 4, 1)
y = torch.squeeze(x)         # shape: (3, 4)
```

# 多项式回归
`多项式回归是回归任务的一种形式，其中自变量和因变量之间的关系是` $M$ `次多项式的一种线性回归形式`

`特征维度为1的多项式表达`

$$
f(\boldsymbol{x};\boldsymbol{w})=w_1x+w_2x^2+...+w_Mx^M+b=\boldsymbol{w}^T\phi(x)+b
$$

其中 $M$ 为多项式的阶数, $\boldsymbol{w}=[w_1,...,w_M]^T$ 为多项式的系数, $\phi(x)=[x,x^2,\cdots,x^M]^T$ 为多项式基函数, 将原始特征 $x$ 映射为 $M$ 维的向量. 

`当特征维度大于1时, 存在不同特征之间交互的情况, 这是线性回归无法实现, 当特征维度为2, 多项式阶数为2时的多项式回归`

$$
f(\boldsymbol{x};\boldsymbol{w})=w_1x_1+w_2x_2+w_3x_1^2+w_4x_1x_2+w_5x_2^2+b
$$

当自变量和自变量之间不是线性关系时, 可以定义非线性基函数对特征进行变换, 从而可以使得线性回归算法实现非线性的曲线拟合

# 封装 Runner 类
`在一个任务上应用机器学习方法的流程基本上包括：数据集构建、模型构建、损失函数定义、优化器、模型训练、模型评价、模型预测等环节`

<div>
  <img src=https://camo.githubusercontent.com/6bb240f44fbc5f40d87e56f0e62f2c37b66ab6de44f5f5d75abb97dea7cfd286/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f66646236353664616164623334396137383536306661343634623064653566613564363334323366633232333461646661633438653666663032306136643630 alt="Runner封装" width="800">
</div>

[All_explain](https://github.com/nndl/practice-in-paddle/blob/main/chap2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%BF%B0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%BF%B0-%E4%B8%8B.ipynb)


# 基于线性回归的波士顿房价预测__实现并组装 Runner 类

## 1 实验流程:
1. 数据处理: 包括数据清洗(缺失值和异常值处理), 数据集划分, 以便数据可以被模型正常读取，并具有良好的泛化性
2. 模型构建: 定义线性回归模型类
3. 训练配置: 训练相关的一些配置，如：优化算法、评价指标等
4. 组装训练框架Runner: `Runner`用于管理模型训练和测试过程
5. 模型训练和测试: 利用`Runner`进行模型训练和测试

## 2 数据处理
`共506条样本数据，每条样本包含了12种可能影响房价的因素和该类房屋价格的中位数,各字段含义`

| 字段名   | 类型  | 含义                                     |
|----------|-------|------------------------------------------|
| CRIM     | float | 该镇的人均犯罪率                         |
| ZN       | float | 占地面积超过25,000平方呎的住宅用地比例   |
| INDUS    | float | 非零售商业用地比例                       |
| CHAS     | int   | 是否邻近 Charles River（1=邻近；0=不邻近）|
| NOX      | float | 一氧化氮浓度                             |
| RM       | float | 每栋房屋的平均客房数                     |
| AGE      | float | 1940年之前建成的自用单位比例             |
| DIS      | float | 到波士顿5个就业中心的加权距离            |
| RAD      | int   | 到径向公路的可达性指数                   |
| TAX      | int   | 全值财产税率                             |
| PTRATIO  | float | 学生与教师的比例                         |
| LSTAT    | float | 低收入人群占比                           |
| MEDV     | float | 同类房屋价格的中位数                     |

## 绘制箱形图

<div>
  <img src="https://camo.githubusercontent.com/6b2be8bff95cb428080ac9699ca0e65a639c5e57dc4a56e8032a3cddae91b8b2/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f32343233353062626436306434616662626564633632626463353537396135346661386533373564323438313430386439626537373036396535373866333965" alt="箱形图" width="800">
</div>

`上四分位数(Q3): 是 第 75 百分位 数, 表示有 75% 数据 <= Q3, 25% 的数据 > Q3`

`中位数(Q2): 是 第 50 百分位数, 即数据的中间值`

`下四分位数(Q1): 是 第 25 百分位数`

它和均值无关