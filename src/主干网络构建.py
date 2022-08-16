import random
import  pandas as pd
import  numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import os
import copy

def open_csv(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    readbook = pd.read_csv(f'{filename}.csv')
    nplist = readbook.T.to_numpy()
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
split = 0.3  # 测试集占数据集整体的多少
ifshuffle = 0  # 1为打乱数据集，0为不打乱
feature, label = open_csv('数据集')
x_train, x_test, y_train, y_test = split_data_set(feature, label, split, ifshuffle)
# 将数据转为tensor格式
traininput, trainlabel = inputtotensor(x_train, y_train)
testinput, testlabel = inputtotensor(x_test, y_test)
#数据已归一化
print(traininput)
print(trainlabel)

# 定义一个简单的数据集
class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

train_ds = tabularDataset(traininput, trainlabel)
test_ds = tabularDataset(testinput,testlabel)
#train_ds = TensorDataset(traininput, trainlabel)
#test_ds  = TensorDataset(testinput,testlabel)

class tabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4608, 500)
        self.lin2 = nn.Linear(500, 20)
        self.lin3 = nn.Linear(20,2) #最后输出为2，因为是分类问题
        #self.lin4 = nn.Linear(20,2)
        self.bn_in = nn.BatchNorm1d(4608)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(20)
       # self.bn3 = nn.BatchNorm1d(20)


    def forward(self, x_in):
        # print(x_in.shape)
        x = self.bn_in(x_in)

        x = F.relu(self.lin1(x))
        x = self.bn1(x)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)

        #x = F.relu(self.lin3(x))
        #x = self.bn3(x)

        x = nn.functional.softmax(self.lin3(x), dim=1)
        #x = self.lin3(x)
        return x

'''
class Batch_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(4608, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(1000, 2),
            )

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out
'''

#训练前指定使用的设备
DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda")
print(DEVICE)

#损失函数
criterion =nn.CrossEntropyLoss()

#实例化模型
model = tabularModel().to(DEVICE)
print(model)

#学习率
LEARNING_RATE=0.001
#批大小，决定一个epoch有多少个Iteration
batch_size = 28
#优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#DataLoader加载数据，shuffle表示每一个 epoch是否为乱序
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_ds,batch_size=batch_size)

#训练100轮，所有的训练样本输入到模型中称为一个epoch
TOTAL_EPOCHS=80

#开始训练
for epoch in range(TOTAL_EPOCHS):
    # 记录损失函数
    losses = [];
    for i, (x, y) in enumerate(train_dl):
        model.train()
        x = x.float().to(DEVICE) #输入必须为float类型
        y = y.long().to(DEVICE) #结果标签必须为long类型
        #向前传播
        outputs = model(x)
        #计算损失函数
        loss = criterion(outputs, y)
        #清空上一轮的梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #参数更新
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(losses)))

    # 测试准确率
    model.eval()
    correct = 0
    total = 0
    for j, (x, y) in enumerate(test_dl):
        x = x.float().to(DEVICE)
        y = y.long()
        outputs = model(x).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()
    print('准确率: %.4f %%' % (100 * correct / total))


