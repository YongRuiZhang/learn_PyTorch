# -*- coding: utf-8 -*-
# @FileName  :train.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 19:59
# @Aim       :完整走一次训练的流程
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *  # 3.引入模型

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

# 4.创建网络模型
cifar10_model = CIFAR10_Model()

# 5.设置使用的损失函数
loss_fn = nn.CrossEntropyLoss()

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
writer = SummaryWriter(log_dir='./logs')

# 8. 训练
for i in range(epoch):
    print('Epoch {}/{} begin...'.format(i + 1, epoch))
    # 训练步骤开始
    cifar10_model.train()  # 将模型修改为训练模式，作用很小，只对BatchNormal和Dropout有用（这个网络模型中没有用）
    for images, targets in train_data_loader:
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
            print('训练次数:{}, Loss: {:.4f}'.format(total_train_step, loss.item()))  # 注意loss是一个tensor类型的，loss.item()才是值
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始，在torch.no_grad()中就不会改变梯度了
    cifar10_model.eval()  # 将模型修改为测试模式，同样作用很小，只对BatchNormal和Dropout有用（这个网络模型中没有用）
    total_test_loss = 0  # 测试集整体损失
    total_accuracy = 0  # 测试集预测正确个数
    with torch.no_grad():
        for images, targets in test_data_loader:
            output = cifar10_model(images)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()

            pred = output.argmax(1)  # 求横向最大值，参数为0则是求纵向的最大值。可以认为最大值就是预测类
            accuracy = torch.sum(pred == targets).item()  # 预测正确的个数
            total_accuracy += accuracy  # 总体预测正确个数

    total_test_step += 1
    print('Epoch {}/{}, Test Loss: {:.4f}, Accuracy: {}'.format(i + 1, epoch, total_test_loss,
                                                                accuracy / total_test_step))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)

    # 10. 保存每一轮的模型
    torch.save(cifar10_model, './model/cifar10_model_{}.pth'.format(i))
    # torch.save(cifar10_model.state_dict(), './cifar10_model_{}.pth'.format(i))  # 官方推荐的保存方式
    print('模型已保存')

# 9. 试用了TensorBoard要记得关闭
writer.close()
