# -*- coding: utf-8 -*-
# @FileName  :06_nn_optim.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 16:23
# @Aim       :在上一个文件求得反向传播梯度的基础上，使用优化器更新权重参数
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 获取数据集
dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 处理数据集
dataloader = DataLoader(dataset, batch_size=1)


# 在文件 03_搭建实战和Sequential.ipynb中搭建的模型
class MY_CIFAR10(nn.Module):
    def __init__(self):
        super(MY_CIFAR10, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 使用交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 实例化模型
my_model = MY_CIFAR10()

# 使用随机梯度下降优化器
optim = torch.optim.SGD(my_model.parameters(), lr=0.01)  # 注意第一个参数是 “模型的参数”因此初始化优化器要在实例化模型之后

# 进行20轮的学习,运行结果可以看出running_loss在不断减小
for epoch in range(20):
    running_loss = 0.0  # 用于记录在当前这轮学习中的整体损失误差的求和
    for data in dataloader:
        imgs, targets = data
        outputs = my_model(imgs)
        result_loss = loss(outputs, targets)

        # 给下面三行代码打上断点Debug，用和上一个文件中相同的方式找到weight，此时要关注grad和data（grad为反向传播计算的梯度，data为实际的权重参数）
        optim.zero_grad()  # 在用优化器调整前，先把模型的中的梯度先调整为0，因为上一次更新的梯度对于这次计算是没有用的（第二次运行到这行时关注grad
        result_loss.backward()  # 利用反向传播，求得每个节点梯度（运行到这行关注grad
        optim.step()  # 对模型参数进行调优（运行到这行时关注data
        running_loss = running_loss + result_loss
    print('Epoch: {}/{}.. ,running_loss:{}'.format(epoch + 1, 20, running_loss))