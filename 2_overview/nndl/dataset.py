import torch
from matplotlib import pyplot as plt

torch.manual_seed(42)

def linear_func(x, w=1.2, b=0.5):
    y = w*x + b
    return y

def create_toy_data(func, interval, sample_num, noise = 0.0, add_outlier = False, outlier_ratio = 0.001):
    """
    根据给定的函数，生成样本
    输入：
       - func 函数
       - interval x的取值范围
       - sample_num 样本数目
       - noise 噪声均方差
       - add_outlier是否生成异常值
       - outlier_ratio异常值占比
    输出：
       - X: 特征数据 shape=[n_samples,1]
       - y: 标签数据 shape=[n_samples,1]

    用均匀分布模拟数据 | 用高斯分布模拟噪声
       - torch.rand 生成 均匀分布 的随机数,常用于 均匀采样输入数据（如 $x$），以确保覆盖整个输入区间,避免偏差。  
       - torch.normal 生成 高斯（正态）分布 的随机数,常用于 模拟标签噪声,因为现实中的测量误差通常由多种微小因素叠加而成,近似服从高斯分布,且其数学性质优良(如与最小二乘损失对应)
       ✅ 均匀采样输入 → 全面探索；高斯噪声标签 → 贴近现实、理论合理
    """
    # 均匀采样
    # 使用 torch.rand 在生成 sample_num个随机数
    X = torch.rand(sample_num) * (interval[1] - interval[0]) + interval[0]
    y = func(X)

    # 生成高斯分布的标签噪声
    # 使用 torch.normal 生成 0 均值, noise标准差的数据
    epsilon = torch.normal(mean=0.0, std=noise, size=y.shape)
    y = y + epsilon

    if add_outlier: # 生成额外的异常点
        outlier_num = int(len(y)*outlier_ratio)
        if outlier_num > 0:
            # 使用 torch.randint生成服从均匀分布, 范围在[0, len(y)] 的随机 Tensor
            outlier_idx = torch.randint(0, len(y), (outlier_num,))
            y[outlier_idx] = y[outlier_idx] * 5
    
    return X, y


if __name__ == "__main__":
    func = linear_func
    interval = (-10, 10)
    train_num = 100
    test_num = 50
    noise = 2
    X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise, add_outlier=False)
    X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise, add_outlier=False)

    # X_train_large, y_train_large = create_toy_data(func=func, interval=interval, sample_num=5000, noise=noise, add_outlier=False)

    X_underlying = torch.linspace(interval[0], interval[1], train_num)
    y_underlying = linear_func(X_underlying)

    plt.scatter(X_train, y_train, marker='*', facecolor="none", edgecolors='#e4007f', s=50, label="train data")
    plt.scatter(X_test, y_test, facecolor="none", edgecolor='#f19ec2', s=50, label="test data")
    plt.plot(X_underlying, y_underlying, c='#000000', label=r"underlying distribution")
    plt.legend(fontsize='x-large') # 给图像加图例
    plt.savefig('ml-vis.png')
    plt.show()