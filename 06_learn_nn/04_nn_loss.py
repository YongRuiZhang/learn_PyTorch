# -*- coding: utf-8 -*-
# @FileName  :04_nn_loss.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 15:34
# @Aim       :介绍几种简单的，常用的损失函数，以及如何使用
import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 对应位置相减取绝对值，然后求平均值或求和（默认为求平均值）
loss = nn.L1Loss()  # 求平均值
result = loss(inputs, targets)
loss2 = nn.L1Loss(reduction='sum')  # 求和
result2 = loss2(inputs, targets)
print(result, result2)

# 对应位置相减求平方，然后求平均值或求和（默认求平均值）
loss_mse = nn.MSELoss()
result3 = loss_mse(inputs, targets)
print(result3)

# 交叉熵损失函数：假设有三个类别：[person, dog, cat]，则其对应的下标分别为[0, 1, 2]
# 现在有一组output为[0.1, 0.2, 0.3]，即x；target为1（表示target下标为1，即dog），即class
# 则计算公式：Loss(x, class) = -x[class] + log(\sum\limits_{j} exp(x[j])) = -0.2 + ln(exp(0.1) + exp(0.2) + exp(0.3))
x = torch.tensor([0.1, 0.2, 0.3])
targets = torch.tensor([1])

x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x, targets)
print(result)