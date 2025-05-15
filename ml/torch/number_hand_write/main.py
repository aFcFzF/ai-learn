
'''
@file main.py
@author afcfzf(9301462@qq.com)
'''

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 4个全连接层
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 先做全连接线性计算，套一个激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 导入数据
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 第一个参数是当前目录
    data_set = MNIST("./data", is_train, transform=to_tensor, download=True)
    # 第1个批次15张图片
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1

    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练轮次
    for epoch in range(2):
        # pytorch模板代码
        for (x, y) in train_data:
            # 初始化
            net.zero_grad()
            # 正向传播
            output = net.forward(x.view(-1, 28*28))
            # 计算损失
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向误差传播
            loss.backward()
            # 优化网络参数
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()