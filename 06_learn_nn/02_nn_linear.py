# -*- coding: utf-8 -*-
# @FileName  :02_nn_linear.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/30 16:28
# @Aim       :使用线性层，先用reshape将图片拉成一维度（模拟卷积工作），再利用线性层映射
import torch
import torchvision.datasets
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


# define linear module
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)  # 将196608变为10

    def forward(self, x):
        output1 = self.linear1(x)
        return output1


# create linear module
linear_module = MyModule()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # original image size： batch=64，channel=3，height=32，weight=32

    # output = torch.reshape(imgs, (1, 1, 1, -1))  # use the reshape function to make the image linear
    output = torch.flatten(imgs)  # use the flatten function to make the image linear
    print(output.shape)  # reshaped image size： batch=1，channel=1，height=1，weight=196608

    # use linear module to reduce weight from 196608 to 10
    output = linear_module(output)
    print(output.shape)  # linear image size: batch=1，channel=1，height=1，weight=10
