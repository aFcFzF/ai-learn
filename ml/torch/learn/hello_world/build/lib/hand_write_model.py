'''
@file hand_write.py
@author afcfzf(9301462@qq.com)
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# 训练/测试资料的维度
# print(train_ds.data.shape, test_ds.data.shape)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(256, 10)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, x):
        # 完全连接层 + dropout + 完全连接层 + dropout + log_softmax
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # dropout 每次训练时，随机丢弃一些神经元，避免过拟合
        # 只在训练时运作，预测时会忽略Dropout，不会有任何作用
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

model = Net().to(device)
