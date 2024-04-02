# @FileName  :read_data.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 16:29

# 导入 Dataset 模块
from torch.utils.data import Dataset
# 导入Image用于读取图片（安装cv2有点慢）
from PIL import Image
import os


# 新建一个类，继承 Dataset 类
class MyData(Dataset):
    # 初始化
    # root_dir 为数据集的根目录 （dataset/train）
    # label_dir 为标签目录 （ants）
    def __init__(self, root_dir: object, label_dir: object) -> object:
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # 结果为 dataset/train/ants
        self.img_path = os.listdir(self.path)

    # 获取每一个数据（图片）
    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    # 定义数据集长度
    def __len__(self):
        return len(self.img_path)


root_dir = '../dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)  # 蚂蚁数据集
bees_dataset = MyData(root_dir, bees_label_dir)  # 蜜蜂数据集

# 合并数据集
train_dataset = ants_dataset + bees_dataset
