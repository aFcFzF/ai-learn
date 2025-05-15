'''
@file train.py
@author afcfzf(9301462@qq.com)
'''

from hand_write_model import *
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

epochs = 5
lr=0.1

PATH_DATASETS = "" # 预设路径

BATCH_SIZE = 1024  # 批量

# 下载 MNIST 手写阿拉伯数字 训练资料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=600)

# TODO: 这里什么意思 adam 是最常用的优化器：使用大多数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []

for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 计算损失
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = 100 * batch_idx / len(train_loader)
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')

plt.plot(loss_list, 'r')
plt.show()
