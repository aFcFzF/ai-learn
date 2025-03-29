import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
# TODO: 没接触过
# from torchmetrics import Accuracy
# TODO: 没接触过
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

PATH_DATASETS = "./data"
BATCH_SIZE = 1024
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)

# 下载训练数据
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())

# 下载测试数据
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())

# 训练/测试数据维度
# print(train_ds.data.shape, test_ds.data.shape)

# print(train_ds.targets[0:10])

# print(train_ds.data[0])

# 获取第1张28x28，转成hotpot
data = train_ds.data[0]
data[data > 0] = 1
data = data.numpy()
# print(data)
# print(data.shape)
# plt.imshow(data)
# plt.show()

# img = str()
# for i in range(data.shape[0]):
#     row = ''.join(str(data[i]))
#     img += (row + '\n')

# print(isinstance(img, str))


model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 256),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 10)
).to(device)

optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

model.train()

loss_list = []

epochs = 5

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=600)

for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
#         if batch_idx == 0 and epoch == 1: print(data[0])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')

plt.plot(loss_list, 'r')
plt.show()