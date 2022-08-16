import  numpy as np
import torch.nn as nn
import torch
import random
import  pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import ExtraTreesClassifier

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


def inputtotensor(inputtensor, labeltensor):
    """
    将数据集的输入和标签转为tensor格式
    :param inputtensor: 数据集输入
    :param labeltensor: 数据集标签
    :return: 输入tensor，标签tensor
    """
    inputtensor = np.array(inputtensor)
    inputtensor = torch.tensor(inputtensor)

    labeltensor = np.array(labeltensor)
    labeltensor = labeltensor.astype(float)
    labeltensor = torch.LongTensor(labeltensor)

    return inputtensor, labeltensor


# 数据划分为训练集和测试集和是否打乱数据集
split = 0.1 # 测试集占数据集整体的多少
ifshuffle = 1  # 1为打乱数据集，0为不打乱
feature1, label1 = open_csv('Escherchia')
feature2, label2 = open_csv('选择特征2')
x_train1, x_test1, y_train1, y_test1 = split_data_set(feature1, label1, split, ifshuffle)
x_train2, x_test2, y_train2, y_test2 = split_data_set(feature2, label2, split, ifshuffle)

# 函数softmax
def softmax(input_W, input_x, input_b):
    sum = []  # 储存每次sim的值
    for i in range(len(input_W)):
        result = sim(input_W[i], input_x) + input_b
        sum.append(result)
    #sum = np.array(sum)
    #sum = sum.reshape(-1,1)  # 向量转置
    sum = torch.tensor(sum)
    sum = nn.functional.softmax(sum,dim=0)
    sum = sum.numpy()
    loc = np.argmax(sum)
    return loc #返回最大值位置


# 函数sim
def sim(input_w, input_x):
    input_x = np.transpose(input_x)  # 矩阵转置
    result1 = np.dot(input_w, input_x) #矩阵乘法

    w_norm = np.linalg.norm(input_w, ord=2, axis=False, keepdims=False)  # 求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=False, keepdims=False)
    result2 = w_norm * x_norm
    return result1 / result2

def test(input_W, input_X,Label_W,Label_X):
    correct=0
    for i in range(len(input_X)):
        loc=softmax(input_W,input_X[i],0)
        if(Label_W[loc]==Label_X[i]):
            correct=correct+1
        print(correct)
    print('准确率: %.4f %%' % (100 * correct / len(input_X)))


test(x_train2,x_test2,y_train2,y_test2)
