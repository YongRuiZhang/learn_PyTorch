# @FileName  :Dataloader.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/26 18:50

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor())

# 每次取从 test_data 中取四个数据（相当于取test_data[0]、test_data[1]、test_data[2]、test_data[3]）并打包（返回imgs(img0、img1、img2、img3)和targets），
# 并打乱（洗牌），加载在主进程，
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 查看数据集第一个元素图片大小和target
img, target = test_data[0]
print(img.shape)  # torch.Size([3, 32, 32]) ,表示三通道（彩色），高度32，宽度32的图片
print(target)  # 3

writer = SummaryWriter("../04_learn_useTorchvisionDataset/logs")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
