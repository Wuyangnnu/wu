import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 函数softmax
def a(input_W, input_X, input_b):
    num = []  # 储存每次sim的值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_X) + input_b
        num.append(result)
        T = F.softmax()
        print(num)
    return max(num) #返回最大值


# 函数sim
def sim(input_w, input_x):
    #input_w = np.transpose(input_w)  # 矩阵转置
    result1 = np.dot(input_w, input_x) #矩阵乘法
    w_norm = np.linalg.norm(input_w, ord=2, axis=False, keepdims=False)  # 求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=False, keepdims=False)
    result2 = w_norm * x_norm
    return result1 / result2

x=np.array([[1,3,5,7,9], [2,4,6,8,10],[1,2,3,4,5]])
y=np.array([1,2,3,4,5])
y=np.exp(y)
y=y.reshape(-1,1)
result = a(x,y,0)
#T=F.softmax(result)
print(x)
print(y)
print(T)



