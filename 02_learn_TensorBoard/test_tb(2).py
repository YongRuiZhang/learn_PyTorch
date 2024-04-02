# @FileName  :test_tb(2).py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 20:50
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

wirter = SummaryWriter("logs")

img_path = "/Users/yonruizhang/MyUse/notes/pytorch/learn-pytorch/练手数据集/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

wirter.add_image("test_image", img_array, 1, dataformats='HWC')

wirter.close()