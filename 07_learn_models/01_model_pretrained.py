# -*- coding: utf-8 -*-
# @FileName  :01_model_pretrained.py
# @Author    :632107110111_张永锐
# @Time      :2024/3/31 16:44
# @Aim       :使用PyTorch自带的vgg16模型，vgg16官方文档链接：https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16
# @Notice    :使用这个代码需要下载vgg16的pth文件，比较大，尽量少运行该代码
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 获取数据集，文档中提示要使用ImageNet数据集（使用ImageNet数据集链接：https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet）
# train_data = torchvision.datasets.ImageNet(root='../data_image_net', split="train", download=True, transform=torchvision.transforms.ToTensor())  # 注意：使用就会报错，因为数据集太大了，已经不给公开访问了

# 改版了，现在的版本不是这样使用的，了解即可
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 只有网络模型，没有初始化参数
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 既有网络模型，又有初始化参数


# ======== 增加层 ========
print(vgg16_true)
# 给vgg16加一层线性层
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
# 给vgg16中的classifier加一层线性层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# ======= 修改层 ========
print(vgg16_false)
# 将vgg16中的classifier的第7层修改为in_feature为4096，out_feature为10的线性层
vgg16_false.classifier[6] = nn.Linear(4096, 10)

# ======= 保存模型 =======
# 保存方式一  使用这种方式一定要让程序本身能知道模型的结构（用库里的模型不用考虑，自己写的模型就要import或者在前面定义）
torch.save(vgg16_false, './vgg16_false.pth')  # 保存模型结构和参数
loadedModel1 = torch.load('./vgg16_false.pth', map_location=torch.device('cpu'))  # 加载模型和参数，注意，下载的模型可能是在cuda上训练的，如果用cpu环境就会报错，因此要使用map_location设置本地的环境
print(loadedModel1)
# 保存方式二  用字典的形式保存参数状态（官方推荐，保存后文件更小）
torch.save(vgg16_false.state_dict(), './vgg16_false2.pth')  # 保存模型参数
loadedModel2 = torch.load('./vgg16_false2.pth')  # 只有模型参数，没有模型结构
print(loadedModel2)
vgg16_false2 = torchvision.models.vgg16(pretrained=False)
vgg16_false2.load_state_dict(loadedModel2)  # 正确方式是先加载模型结构，然后用模型结构的load_state_dict函数，给定参数为模型参数
print(vgg16_false2)

