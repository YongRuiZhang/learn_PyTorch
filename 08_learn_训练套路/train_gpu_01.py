# -*- coding: utf-8 -*-
# @FileName  :train_gpu_01.py
# @Author    :632107110111_张永锐
# @Time      :2024/4/1 10:38
# @Aim       :在GPU上训练的第一种方式
# @Method    :将1.网络模型；2.损失函数；3.数据（输入，标注）转移到cuda上即可。就是对这三个变量调用cuda()，注意看本文件的步骤11
# @Notice    :其中比较麻烦的是数据转移到cuda上，是要在训练和测试的循环中，分别将images和targets转移到cuda上
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

start_time = time.time()

# from model import *  # 3.引入模型，这里直接把网络模型放在下面，因为用GPU跑的时候不在本地，写在一个文件中方便一点

# 1.准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
train_data_size = len(train_data)
test_data_size = len(test_data)

# 2.利用DataLoader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)


# 3.搭建神经网络模型 (前面搭建过的CIFAR10模型)，但是为了规范，一般都会放到一个单独的文件中，这里放到model文件中
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


# 查看是否能使用GPU
print(torch.cuda.device_count(), torch.cuda.is_available())

# 4.创建网络模型
cifar10_model = CIFAR10_Model()
# 11.1 网络模型转移到cuda上
if torch.cuda.is_available():
    cifar10_model = cifar10_model.cuda()  # 可以直接写成cifar10_model.cuda()，不用再赋值回去


# 5.设置使用的损失函数
loss_fn = nn.CrossEntropyLoss()
# 11.2 损失函数转移到cuda上
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 可以直接写成 loss_fn.cuda(),不用赋值回去

# 6.设置优化器
learning_rate = 1e-2  # 最好把学习率定义为一个变量
optimizer = torch.optim.SGD(cifar10_model.parameters(), lr=learning_rate)

# 7.记录训练网络的一些参数
# 记录网络训练的次数
total_train_step = 0
# 记录网络测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 9. 添加TensorBoard记录训练的损失值，如下设置后，运行可以看到test_loss和train_loss
writer = SummaryWriter(log_dir='./logs_gpu_01')

# 8. 训练
for i in range(epoch):
    print('Epoch {}/{} begin...'.format(i + 1, epoch))
    # 训练步骤开始
    cifar10_model.train()  # 将模型修改为训练模式，作用很小，只对BatchNormal和Dropout有用（这个网络模型中没有用）
    for images, targets in train_data_loader:
        # 11.3.1 将训练数据转移到cuda上
        if torch.cuda.is_available():
            images = images.cuda()  # 数据一定要赋值回去
            targets = targets.cuda()
        # 8.1 将图片作为输入放到网络中进行计算，得到输出
        output = cifar10_model(images)

        # 8.2 计算损失函数值
        loss = loss_fn(output, targets)

        # 8.3 优化参数
        # 8.3.1 梯度清零
        optimizer.zero_grad()
        # 8.3.2 方向传播计算梯度
        loss.backward()
        # 8.3.3 优化参数
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:  # 减少输出，每一步都输出的话输出量太大了
            end_time = time.time()
            print(end_time - start_time)
            print('训练次数:{}, Loss: {:.4f}'.format(total_train_step, loss.item()))  # 注意loss是一个tensor类型的，loss.item()才是值
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始，在torch.no_grad()中就不会改变梯度了
    cifar10_model.eval()  # 将模型修改为测试模式，同样作用很小，只对BatchNormal和Dropout有用（这个网络模型中没有用）
    total_test_loss = 0  # 测试集整体损失
    total_accuracy = 0  # 测试集预测正确个数
    with torch.no_grad():
        for images, targets in test_data_loader:
            # 11.3.2 将测试数据转移到cuda上
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
            output = cifar10_model(images)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()

            pred = output.argmax(1)  # 求横向最大值，参数为0则是求纵向的最大值。可以认为最大值就是预测类
            accuracy = torch.sum(pred == targets).item()  # 预测正确的个数
            total_accuracy += accuracy  # 总体预测正确个数

    total_test_step += 1
    print('Epoch {}/{}, Test Loss: {:.4f}, Accuracy: {}'.format(i + 1, epoch, total_test_loss,
                                                                accuracy / total_test_step))
    writer.add_scalar('test_loss', total_test_loss, test_data_size)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)

    # 10. 保存每一轮的模型
    torch.save(cifar10_model, './model/cifar10_model_{}.pth'.format(i))
    # torch.save(cifar10_model.state_dict(), './cifar10_model_{}.pth'.format(i))  # 官方推荐的保存方式
    print('模型已保存')

# 9. 试用了TensorBoard要记得关闭
writer.close()
