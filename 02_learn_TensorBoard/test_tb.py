# @FileName  :test_tb.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 17:19

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# writer.add_image()

# 绘制 y = 3x
for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

writer.close()

