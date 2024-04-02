# -*- coding: utf-8 -*-
# @FileName  :model.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 21:08
import torch
from torch import nn


class CIFAR10_Model(nn.Module):
    def __init__(self):
        super(CIFAR10_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 一般写好一个网络后，都要有个main先来测试神经网络正确性
if __name__ == '__main__':
    Cifar10_model = CIFAR10_Model()

    # 一般就是模拟一个输入的数据，主要是构造相同尺寸，然后查看输出是否正确
    input = torch.ones((64, 3, 32, 32))
    output = Cifar10_model(input)
    print(output.shape)
