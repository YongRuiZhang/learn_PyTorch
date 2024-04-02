# -*- coding: utf-8 -*-
# @FileName  :05_nn_loss_network.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 15:59
# @Aim       :在已经搭好的模型中使用CIFAR10数据集计算交叉熵损失函数并计算反向传播，观察梯度的变化（为下一个文件的优化器作铺垫）
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 获取数据集
dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 处理数据集
dataloader = DataLoader(dataset, batch_size=1)


# 在文件搭建实战01.ipynb中搭建的模型
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
for data in dataloader:
    imgs, targets = data
    outputs = my_model(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)  # 输出每一个batch的损失结果，在定义反向传播前的输出形式为：tensor(2.2957, grad_fn=<NllLossBackward0>)
    # 使用反向传播,这里使用调试功能（断点打在52行）来看一下Gradient的变化
    # 进入后先找到并点开模型：my_model；然后找到model1并点开；点开Protected Attributes；点开_modules（里面包含的就是网络中的每一层了）；随便点开一层，找到下面的weight然后点开。
    # 再点击运行到下一行，注意观察grad
    result_loss.backward()  # 注意，一定要用求loss之后的结果这个变量来反向传播
    print("OK")

