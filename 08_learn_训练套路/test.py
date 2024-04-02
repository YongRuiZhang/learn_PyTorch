# -*- coding: utf-8 -*-
# @FileName  :test.py
# @Author    :632107110111_张永锐
# @Time      :2024/4/1 21:39
# @Aim       :利用训练好的模型进行测试，不仅仅是测试，它其实也是一种实际应用，例如predict
import torchvision
from PIL import Image
from model import *

# 1. 测试图片路径
image_path = 'testImages/dog.png'
# image_path = 'testImages/airplane.png'  # 测试飞机，预测结果为0即为正确

# 2. 读入图片
image = Image.open(image_path)  # 此时为PIL类型
image = image.convert('RGB')  # 将图片转换为RGB三通道，png格式的图片可能会除了RGB三通道外还有一个透明化通道
# 3. 转换为Tensor类型，这里利用Compose，参数为多个transforms，这里先Resize，再转换为Tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)

# 4. 加载网络模型和参数，因为我使用的是官方推荐的第二中保存方式，所以要先获得网络模型的结构
Cifar10_model = CIFAR10_Model()  # 模型结构
Cifar10_model.load_state_dict(torch.load('./model/cifar10_gpu_02_model_19.pth'))  # 给模型添加训练好的参数

# 5. 测试
Cifar10_model.eval()  # 要先转换为测试（评估）模式
image = torch.reshape(image, (1, 3, 32, 32))  # 这里BatchSize要提前设置，否则报错
# 在无梯度的环境中测试
with torch.no_grad():
    output = Cifar10_model.forward(image)
# print(output.shape)

# 获得预测结果
pred = output.argmax(1)
# 可以通过训练模型时的debug发现：0-airplane；1-automobile；2-bird；3-cat；4-deer；5-dog；6-frog；7-horse；8-ship；9-truck。因此这里输出5就表示预测正确了
print(pred.item())

