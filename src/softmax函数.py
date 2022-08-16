import  numpy as np
# 函数softmax
def softmax(input_W, input_X, input_b):
    num = []  # 储存每次sim的值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_X) + input_b
        num.append(result)
    return max(num)


# 函数sim
def sim(input_w, input_x):
    input_w = np.transpose(input_w)  # 矩阵转置
    print(input_w)
    result1 = np.dot(input_w, input_x)
    print(result1)
    w_norm = np.linalg.norm(input_w, ord=2, axis=None, keepdims=False)  # 求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=None, keepdims=False)
    result2 = w_norm * x_norm
    return result1 / result2

x=np.array([[1,1,1,1,1], [2,3,3,3,3],[2,1,2,3,3]])
y=np.array([2,2,2,2,2])
#y=y.reshape(-1,1)
print(y)
result = softmax(x,y,0)
print(x)
print(y)
print(result)

#x=np.exp(x)
#y=np.exp(y)
#print(x)
#x=x/np.linalg.norm(x)
#print(x)
