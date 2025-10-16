import math
import copy
import torch
torch.manual_seed(42)

def make_moons(n_samples=1000, shuffle=True, noise=None):
    """
    生成带噪音的弯月形状数据
    输入：
        - n_samples: 数据量大小, 数据类型为 (int)
        - shuffle: 是否打乱数据, 数据类型为 (bool)
        - noise: 以多大的程度增加噪声, 数据类型为 (None或float), noise为None时表示不增加噪声
    输出：
        - X: 特征数据, shape=[n_samples, 2]
        - y: 标签数据, shape=[n_samples]
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # 生成外弧
    outer_circ_x = torch.cos(torch.linspace(0, math.pi, n_samples_out)) # 特征 1 | x 坐标
    outer_circ_y = torch.sin(torch.linspace(0, math.pi, n_samples_out)) # 特征 2 | y 坐标

    # 生成内弧
    inner_cire_x = 1 - torch.cos(torch.linspace(0, math.pi, n_samples_in)) # 特征 1 | x 坐标
    inner_cire_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_samples_in)) # 特征 2 | y 坐标

    print('outer_circ_x.shape:', outer_circ_x.shape, 'outer_circ_y.shape:', outer_circ_y.shape)
    print('inner_circ_x.shape:', inner_cire_x.shape, 'inner_circ_y.shape:', inner_cire_y.shape)

    # 拼接特征
    X = torch.stack(
        [torch.cat([outer_circ_x, inner_cire_x]),
         torch.cat([outer_circ_y, inner_cire_y])],
         dim = 1
    )
    
    print('after concat shape:', torch.cat([outer_circ_x, inner_cire_x]).shape)
    print('X shape:', X.shape)
    
    # 标签
    y = torch.cat([
        torch.zeros(n_samples_out),
        torch.ones(n_samples_in)
    ])

    print('y shape:', y.shape)

    # 打乱
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    # 添加噪声
    if noise is not None:
        X += torch.normal(mean=0.0, std=noise, size=X.shape)

    return X, y

import numpy as np
import torch

def make_multiclass_classification(n_samples=100, n_features=2, n_classes=3, shuffle=True, noise=0.1):
    """
    生成带噪音的多类别数据(PyTorch 版本）
    输入：
        - n_samples:数据量大小,数据类型为int
        - n_features:特征数量,数据类型为int
        - shuffle:是否打乱数据,数据类型为bool
        - noise:以多大的程度增加噪声,数据类型为None或float,noise为None时表示不增加噪声
    输出：
        - X:特征数据,shape=[n_samples, n_features]
        - y:标签数据, shape=[n_samples]
    """
    # 计算每个类别的样本数量
    n_samples_per_class = [int(n_samples / n_classes) for _ in range(n_classes)]
    for i in range(n_samples - sum(n_samples_per_class)):
        n_samples_per_class[i % n_classes] += 1

    # 初始化特征和标签
    X = torch.zeros((n_samples, n_features), dtype=torch.float32)
    y = torch.zeros((n_samples,), dtype=torch.int32)

    # 随机生成 3 个簇中心
    centroids = torch.randperm(2 ** n_features)[:n_classes]
    centroids_bin = np.unpackbits(centroids.numpy().astype('uint8')).reshape((-1, 8))[:, -n_features:]
    centroids = torch.tensor(centroids_bin, dtype=torch.float32)
    centroids = 1.5 * centroids - 1  # 控制簇中心的分离程度

    # 随机生成特征值
    X[:, :n_features] = torch.randn((n_samples, n_features))

    stop = 0
    # 让每个类的特征值集中在对应簇中心附近
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_class[k]
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        # 随机生成线性变换矩阵, 让数据分布更不同
        A = 2 * torch.rand((n_features, n_features)) - 1
        X_k = torch.matmul(X_k, A)
        X_k += centroid # 通过加 让在 原本离得比较开的 几个点开始定义
        X[start:stop, :n_features] = X_k

    # 加噪声
    if noise > 0.0:
        """
        这种“标签扰动”常用于：
            - 分类任务的数据增强
            - 提高模型对噪声数据的鲁棒性
            - 模拟真实世界中标签不准确的情况
            - 比如在一些论文中，这种方法叫：
        Random Label Noise Injection 或 Symmetric Label Noise
        """
        noise_mask = torch.rand((n_samples,)) < noise
        noisy_indices = noise_mask.nonzero(as_tuple=True)[0]
        y[noisy_indices] = torch.randint(0, n_classes, (len(noisy_indices),), dtype=torch.int32)

    # 打乱
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]

    return X, y


torch.manual_seed(42)
# 采样1000个样本
n_samples = 1000
X, y = make_multiclass_classification(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

num_train = 640
num_dev = 160
num_test = 200

X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

# 打印X_train和y_train的维度
print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

# 打印前5个数据的标签
print(y_train[:5])


if __name__ == "__main__":
    # --------------------------------------------------
    # View Data
    # --------------------------------------------------
    """
    X, y = make_moons(n_samples=1000, noise=0.05)
    num_train = 640
    num_dev = 160
    num_test = 200

    X_train, y_train = X[:num_train], y[:num_train]
    X_dev, y_dev = X[num_train:num_train+num_dev], y[num_train:num_train+num_dev]
    X_test, y_test = X[num_train+num_dev:], y[num_train+num_dev:]

    y_train = y_train.reshape([-1,1])
    y_dev = y_dev.reshape([-1,1])
    y_test = y_test.reshape([-1,1])

    print("X_train shape: ", X_train.shape, "y_train shape: y_train.shape")

    print("X_train[:5]: ", X_train[:5])

    print("y_train[:5]: ", y_train[:5])
    import matplotlib.pyplot as plt
    torch.manual_seed(42)
    X, y = make_moons(n_samples=1000, noise=0.05)
    print("Sample X:", X[:5], "Shape X:", X.shape)
    print("Sample y:", y[:5], "Shape y:", y.shape)

    plt.figure(figsize=(5,5))
    plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
    plt.xlim(-3,4)
    plt.ylim(-3,4)
    plt.savefig('linear-dataset-vis.png')
    # plt.show()
    # 训练集 : 测试集 = 8 : 2
    # 训练集 : 验证集 = 8 : 2 
    """

    # -----------------------------------------------------
    # Multi-class data-view
    # -----------------------------------------------------
    import matplotlib.pyplot as plt
    X, y = make_multiclass_classification(n_samples=11, n_features=2, n_classes=3, noise=0.2)
    print("X:", X)
    print("y:", y)

    torch.manual_seed(42)
    # 采样1000个样本
    n_samples = 1000
    X, y = make_multiclass_classification(n_samples=n_samples, n_features=2, n_classes=3, noise=0.2)

    # 可视化生产的数据集，不同颜色代表不同类别
    # plt.figure(figsize=(5,5))
    # plt.scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), marker='*', c=y.tolist())
    # plt.savefig('linear-dataset-vis2.png')
    # plt.show()

    num_train = 640
    num_dev = 160
    num_test = 200

    X_train, y_train = X[:num_train], y[:num_train]
    X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
    X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

    # 打印X_train和y_train的维度
    print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

    # 打印前5个数据的标签
    print(y_train[:5])

