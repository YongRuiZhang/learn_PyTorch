# @FileName  :01_nn_model.py
# @Author    :632107110111_张永锐
# @Time      :2023/11/3 20:43
import torch
import torch.nn as nn


class test_model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


my_model = test_model()
x = torch.tensor(1.0)
output = my_model(x)
print(output)