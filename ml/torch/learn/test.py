import torch as t
from torch.autograd import Variable
import torch.nn as nn

y = t.rand(5, 3)
# print(y)

x = Variable(t.ones(2, 2), requires_grad=True)
y = x.sum()

print(x.grad)
# 反向传播计算
y.backward()

print(x.grad)
x.grad.data.zero_()

y.backward()
print(x.grad)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init()
        # 1 表示输入图片是单通道，6 表示输出通道数, 5 表示卷积核是5x5
        self.conv1 = nn.Conv2d(1, 6, 5) # 卷积后 24 x 24 x 6
        self.pool = nn.AvgPool2d(2, stride=2) # 12 x 12 x 6
        self.conv2 = nn.Conv2d(6, 16, 5) # 8 x 8 x 16
        self.pool = nn.AvgPool2d(2, stride=2) # 池化后  4 x 4 x 16
        # 全连接层, y = wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(t.relu(self.conv1(x)))  # Step 1: 卷积 → 激活 → 池化
        x = self.pool(t.relu(self.conv2(x)))   # Step 2: 卷积 → 激活 → 池化
        x = x.view(-1, 16 * 4 * 4)                    # Step 3: 展平多维特征图
        x = t.relu(self.fc1(x))               # Step 4: 全连接层 → 激活
        x = t.relu(self.fc2(x))               # Step 5: 全连接层 → 激活
        x = self.fc3(x)                          # Step 6: 输出层（无需激活函数）
        return x
