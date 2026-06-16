# 07 - 神经网络基础

## 神经元计算

```text
z = b + w1*x1 + w2*x2 + ...
y = activation(z)
```

## 常见激活函数

- sigmoid
- tanh
- ReLU
- Softmax

## 前向传播

```text
输入 → 隐藏层 → 输出层 → 预测结果
```

## 反向传播

```text
预测结果 → 损失函数 → 计算梯度 → 更新参数
```

参数更新：

```text
W_new = W_old - learning_rate * gradient
```

## 正则化

- L1 / L2；
- Dropout；
- BatchNorm；
- Early Stopping。

## PyTorch 模型模板

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)
```
