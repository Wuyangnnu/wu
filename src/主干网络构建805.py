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
ifshuffle = 1  # 1为打乱数据集，0为不打乱
feature, label = open_csv('数据集3')
feature2 , label2 = open_csv('optimumDataset')
x_train, x_test, y_train, y_test = split_data_set(feature, label, split, ifshuffle)
x_train2,x_test2, y_train2, y_test2 = split_data_set(feature2, label2, 0, 0)

# 将数据转为tensor格式
traininput, trainlabel = inputtotensor(x_train, y_train)
testinput, testlabel = inputtotensor(x_test, y_test)

#对M数据集归一化，并变为二维数组
x_train2 = torch.tensor(x_train2)
x_train2 = nn.functional.normalize(x_train2)#归一化
input_w = x_train2.numpy()

#数据归一化
#traininput=nn.functional.normalize(traininput)
#testinput=nn.functional.normalize(testinput)

#函数softmax
def softmax(input_W,input_X,input_b):

    num = []#储存每次sim的值
    for i in range(len(input_W)):
        result=sim(input_W[i],input_X) + input_b
        num.append(result)
    return  max(num)


#函数sim
def sim(input_w,input_x):
    input_w=np.transpose(input_w)#矩阵转置
    result1 = np.dot(input_x, input_w) #
    w_norm = np.linalg.norm(input_w, ord=2, axis=None, keepdims=False)#求2-范数
    x_norm = np.linalg.norm(input_x, ord=2, axis=None, keepdims=False)
    result2 = w_norm*x_norm
    return  result1/result2



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
        self.lin1 = nn.Linear(1153, 800)
        self.lin2 = nn.Linear(800,78)
        self.lin3 = nn.Linear(78,2) #最后输出为2，因为是分类
        #self.bn_in = nn.BatchNorm1d(1153)
        #self.bn1 = nn.BatchNorm1d(800)
        #self.bn2 = nn.BatchNorm1d(78)


    def forward(self, x_in):
        # print(x_in.shape)
        #x = self.bn_in(x_in)

        x = F.relu(self.lin1(x_in))
        #x = self.bn1(x)

        x = F.relu(self.lin2(x))
        #x = self.bn2(x)
        '''
        x1=x.cpu().detach().numpy()
        x2=[]
        for i in range(len(x1)):
            x2.append(softmax(input_w,x1[i],0))
        x2 = torch.tensor(x2)
        x2= x2.to(torch.device("cuda"))
        print(x2)
        '''
        x = self.lin3(x)
        return x

    #初始化权重和偏置值
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                #nn.init.kaiming_uniform_(m.weight.data)
                #m.bias.data.fill_(0)
                m.bias.data.zero_()

'''
另一种形式
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

#损失函数
criterion =nn.CrossEntropyLoss()

#实例化模型
model = tabularModel().to(DEVICE)
#model.initialize()
print(model)

#学习率
LEARNING_RATE=0.001
#批大小，决定一个epoch有多少个Iteration
batch_size = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


#DataLoader加载数据，shuffle表示每一个 epoch是否为乱序,测试集无需打乱顺序；训练集打乱顺序，为了增加训练模型的泛化能力
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False)

#训练100轮，所有的训练样本输入到模型中称为一个epoch
TOTAL_EPOCHS=100


losses = [];
#开始训练
for epoch in range(TOTAL_EPOCHS):
    model.train()
    # 记录损失函数
    for i, (x, y) in enumerate(train_dl):
        x = x.float().to(DEVICE) #输入必须为float类型
        y = y.long().to(DEVICE) #结果标签必须为long类型
        #向前传播
        outputs = model(x)
        print(outputs)
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
        _, predicted = torch.max(outputs.data, 1)#max函数返回最大值
        total += y.size(0)
        correct += (predicted == y).sum()
    print('准确率: %.4f %%' % (100 * correct / total))


