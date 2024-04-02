# @FileName  :test_transform01.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 21:06

from torchvision import transforms
from PIL import Image

img_path = "/Users/yonruizhang/MyUse/notes/pytorch/learn-pytorch/练手数据集/train/ants_image/0013035.jpg"
img = Image.open(img_path)

tensor_transforms = transforms.ToTensor()
tensor_img = tensor_transforms(img)

print(tensor_img)