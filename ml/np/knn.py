"""
@file knn.py
@author afcfzf(9301462@qq.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import operator

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def createDataset():
    group = np.array([[20, 3], [15, 5], [18, 1], [5, 17], [2, 15], [3, 20]])
    labels = ['服务策略-旅游', '服务策略-洗衣服', '服务策略-收拾家务', '平台策略-照顾婴儿', '平台策略-拍婚纱照', '平台策略-做饭']
    
    return group, labels

group, labels = createDataset()

def classify(in_x: list[int], datas: np.ndarray, labels, k):
    # 将数据重复几遍
    data_size = datas.shape[0]
    diff_mat = np.tile(in_x, (data_size, 1)) - datas
    # 横轴相加 [[1, 2], [3, 4]] = [3, 7]
    distance = (diff_mat ** 2).sum(axis = 1) ** .5
    # 返回的是索引排序 [3, 1, 2] -> [1, 2, 0]
    sort_dist = distance.argsort()

    class_count = {}
    for i in range(k):
        votel_label = labels[sort_dist[i]]
        class_count[votel_label] = class_count.get(votel_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

target = [4, 17];
result = classify(target, group, labels, 3)

print(result)

# draw
x = [item[0] for item in group]
y = [item[1] for item in group]
plt.scatter(x, y, s=30, c='r', marker='o')
for i, val in enumerate(labels):
    plt.text(x[i], y[i], val)

plt.scatter(target[0], target[1], s=50, c='b', marker='X')

plt.show()

