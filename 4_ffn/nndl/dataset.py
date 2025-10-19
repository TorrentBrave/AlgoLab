import math
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


n_samples = 1000
X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.5)

num_train = 640
num_dev = 160
num_test = 200

X_train, y_train = X[:num_train], y[:num_train]
X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

y_train = y_train.reshape([-1,1])
y_dev = y_dev.reshape([-1,1])
y_test = y_test.reshape([-1,1])


if __name__ == "__main__":
    n_samples = 1000
    X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.5)

    num_train = 640
    num_dev = 160
    num_test = 200

    X_train, y_train = X[:num_train], y[:num_train]
    X_dev, y_dev = X[num_train:num_train + num_dev], y[num_train:num_train + num_dev]
    X_test, y_test = X[num_train + num_dev:], y[num_train + num_dev:]

    y_train = y_train.reshape([-1,1])
    y_dev = y_dev.reshape([-1,1])
    y_test = y_test.reshape([-1,1])