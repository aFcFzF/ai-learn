import numpy as np
import matplotlib.pyplot as plt

# 创建一个三维张量
tensor = np.array([
  [
    [1, 2, 0],
    [1, 2, 3]
  ],
  [
    [1, 2, 0],
    [1, 2, 0]
  ],
])

# 显示张量的形状
print("张量形状:", tensor.shape)


# 显示张量的形状
print("张量形状:", tensor.shape)

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 获取三维坐标
x, y, z = np.indices(tensor.shape)

# 绘制三维散点图
ax.scatter(x, y, z, c=tensor.flatten(), cmap='viridis')

# 设置标题和标签
ax.set_title(f"三维张量，形状: {tensor.shape}")
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_zlabel('Z 轴')
plt.show()
