# @FileName  :test01.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 21:26

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "/练手数据集/train/ants_image/0013035.jpg"
img = Image.open(img_path)

tensor_tf = transforms.ToTensor()
tensor_img = tensor_tf(img)

writer.add_image("ToTensor", tensor_img)
writer.close()
