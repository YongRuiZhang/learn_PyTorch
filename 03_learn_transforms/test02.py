# @FileName  :test02.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 21:44

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("/练手数据集/train/ants_image/0013035.jpg")

trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("beforeNormalize", img_tensor)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("afterNormalize", img_norm)
writer.close()

