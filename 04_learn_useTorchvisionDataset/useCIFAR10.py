# @FileName  :useCIFAR10.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 22:32

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="../data", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="../data", train=False, transform=dataset_transform, download=True)

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()