import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

PATH_DATASETS = "" # 预设路径

BATCH_SIZE = 1024  # 批量

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())

X = train_ds.data[0]

plt.imshow(X.reshape(28, 28), cmap='gray')

plt.axis('off')
plt.show()
