import  numpy as np
import torch.nn as nn
import torch
import random
import  pandas as pd
from numpy import *

def open_csv(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    readbook = pd.read_csv(f'{filename}.csv')
   #print(readbook.head()) #数据前五行
   #print(readbook.describe()) #数据结构
   #print(len(readbook)) #数据长度
    nplist = readbook.T.to_numpy()#将 DataFrame 转换为 NumPy 数组
    data = nplist[0:-1].T
    data = np.float64(data)
    target = nplist[-1]
    return data, target

def random_number(data_size, key):
    """
   使用shuffle()打乱
    """
    number_set = []
    for i in range(data_size):
        number_set.append(i)

    if key == 1:
        random.shuffle(number_set)

    return number_set


def split_data_set(data_set, target_set, rate, ifsuf):
    """
    说明：分割数据集，默认数据集的rate是测试集
    :param data_set: 数据集
    :param target_set: 标签集
    :param rate: 测试集所占的比率
    :return: 返回训练集数据、测试集数据、训练集标签、测试集标签
    """
    # 计算训练集的数据个数
    train_size = int((1 - rate) * len(data_set))
    # 随机获得数据的下标
    data_index = random_number(len(data_set), ifsuf)
    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    # 训练集数据
    x_train = data_set[data_index[:train_size]]
    # 测试集数据
    x_test = data_set[data_index[train_size:]]
    # 训练集标签
    y_train = target_set[data_index[:train_size]]
    # 测试集标签
    y_test = target_set[data_index[train_size:]]
    return x_train, x_test, y_train, y_test

def maxminnorm(array): #归一化
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

split = 0.2 # 测试集占数据集整体的多少
ifshuffle = 0  # 1为打乱数据集，0为不打乱
feature1, label1 = open_csv('权值数据集w') #均匀E
feature2, label2 = open_csv('输入测试集X0') #均匀E16552
feature3, label3 = open_csv('VibrioN169610') #含0数据集
W_x, xtest1, W_y, ytest1 = split_data_set(feature1, label1, 0, 0)#数据集M作为权值
X_input, xtest2, Y_input, ytest2 = split_data_set(feature2, label2, 0, 0)#输入数据集X作为测试
X_pre , xtest3 , Y_pre, ytest3 = split_data_set(feature3, label3, 0, 0)
W_x = maxminnorm(W_x)
X_input = maxminnorm(X_input)
X_pre = maxminnorm(X_pre)

# 函数softmax1，训练用，返回值为数组P，即各个分类的概率分布
def softmax1(input_W, input_x, input_b,W_label):
    sum = []   # 储存sim值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_x) + input_b #计算相似度
        sum.append(result)
    sum = torch.tensor(sum)
    sum = nn.functional.softmax(sum,dim=0)
    sum = sum.numpy()
    return sum #返回

# 函数softmax2，测试用，返回相似度最大的向量位置
def softmax2(input_W, input_x, input_b):
    sum = []  # 储存每次sim的值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_x) + input_b
        sum.append(result)
    sum = torch.tensor(sum)
    sum = nn.functional.softmax(sum,dim=0)
    sum = sum.numpy()
    loc = np.argmax(sum)
    return loc #返回最大值位置

# 函数sim
def sim(input_w, input_x):
    input_x = input_x.reshape(-1,1)  # 变为列向量
    input_w = input_w.reshape(1,-1)  # 变为行向量
    temp = np.dot(input_w, input_x) #矩阵乘法
    result1=temp[0][0]
    w_norm = np.linalg.norm(input_w, ord=2, axis=None, keepdims=False)  # 求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=None, keepdims=False)
    result2 = w_norm * x_norm
    return result1 / result2

def SGD(x, y, w, alpha, max_iteration,W_label):
    """
    随机梯度下降法:stochastic_Gradient_Descent
    :param x:train_data
    :param y:train_label
    :param w:初始化权重
    :param alpha:学习速率
    :param max_iteration:迭代次数
    :return:更新后的权值
    """
    num =int(len(w) / 2)
    for j in range(0,num):
        print("第"+str(j)+"组：")
        group = []
        group_y = [0,1]
        group.append(w[num + j])
        group.append(w[j])
        group=np.array(group)
        group_y =np.array(group_y)
        for i in range(0, max_iteration):
            resultList = i%len(X_input)
            index = int(y[resultList])
            P = softmax1(group, x[resultList], 0 ,group_y) #这里得到P里包含K个种类的概率分布
            Y = index
            Y = torch.tensor(Y)
            P = torch.tensor(P)
            loss = nn.CrossEntropyLoss()  # 损失函数
            out = loss(P,Y)  #计算Y与P之间的不相似性
            # 下降梯度
            Y = np.zeros(2)
            Y[index]=1
            P = np.array(P)
            temp = Y-P
            temp = np.array(temp)
            temp2 = np.array(x[resultList])
            temp2 = temp2.reshape(1,-1) #行向量
            temp = temp.reshape(-1,1) #列向量
            gradient = np.dot(temp, temp2)  #计算梯度
            group = group - alpha * gradient
            out_loss = out.tolist() #损失
            print("损失LOSS："+ str(out_loss) )
        print("优化完成")
        w[j] = group[0]
        w[num + j] = group[1]
    return w


def main():
    train_data = X_input
    train_label = Y_input
    w = W_x  # 初始化权重ｗ
    max_iteration =len(X_input) # 迭代次数
    alpha = 0.004# 学习速率,全0：0.004.全1:0。04
    w = SGD(train_data, train_label, w, alpha, max_iteration,W_y)
    return w

def test(input_W, input_X,Label_W,Label_X):
    correct=0
    for i in range(len(input_X)):
        loc=softmax2(input_W,input_X[i],0)
        if(Label_W[loc]==Label_X[i]):
            print(Label_W[loc])
            print(Label_X[i])
            correct=correct+1
        print(correct)
    print('准确率: %.4f %%' % (100 * correct / len(input_X)))


if __name__ == "__main__":
    #test(W_x, X_pre, W_y, Y_pre)
    w = main()
    test(w,X_pre,W_y,Y_pre)