import numpy as np
import torch.nn as nn
import torch


# 函数softmax
def softmax(input_W, input_X, input_b):
    num = []  # 储存每次sim的值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_X) + input_b
        num.append(result)
    num = np.array(num)
    num = num.reshape(-1, 1)  # 向量转置
    num = torch.tensor(num)
    num = nn.functional.softmax(num, dim=0)
    num = num.numpy()
    print(num)
    return max(num)  # 返回最大值


# 定义LOSS函数
criterion = nn.CrossEntropyLoss()


def sgd_updata(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data


# 函数sim
def sim(input_w, input_x):
    input_x = np.transpose(input_x)  # 矩阵转置
    result1 = np.dot(input_w, input_x)  # 矩阵乘法

    w_norm = np.linalg.norm(input_w, ord=2, axis=False, keepdims=False)  # 求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=False, keepdims=False)
    result2 = w_norm * x_norm
    return result1 / result2




x = np.array([[1, 8, 3, 16, 5], [2, 4, 6, 8, 10], [1, 20, 5, 31, 9]])
y = np.array([2, 4, 6, 4, 5])

result = softmax(x, y, 0)
print(x)
print(y)
print(result)

