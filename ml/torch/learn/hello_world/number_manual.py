'''
@file number_manual.py
@author afcfzf(9301462@qq.com)
'''

import numpy as np

# 0为中心正态分布权重矩阵
def gen_weights(n_input: int, neurons: int):
    return np.random.randn(n_input, neurons)

# 激活函数
def reLu(n: float):
    return max(0, n)

# 矩阵点乘
def dot(a, b):
    result = []

    for a_row in a:
        item = []
        dot_sum = 0
        for b_row_idx in range(len(b[0])):
            for a_col_idx, a_col_val in enumerate(a_row):
                dot_sum += (a_col_val * b[a_col_idx][b_row_idx])
                if (a_col_idx == len(a_row) - 1):
                    item.append(dot_sum)
                    dot_sum = 0
            if (b_row_idx == len(b[0]) - 1):
                result.append(item)

    return result

# 全连接
img_data = [
    [1, 2, 3],
    [2, 1, 0]
]

b = [[-0.67123784, -3.13438703],
       [ 1.1546372 , -0.79405653],
       [ 0.80232736,  0.41167853]]

print(dot(img_data, b))

# y = a * W(T) + b  入参是像素，出参是神经元
def linear(src: int, target: int):
    src_size = int(src ** .5)
    w = gen_weights(src_size, int(target ** 0.5))
    return np.array(np.random.rand(src_size, src_size)).dot(w)

print('-----')
print(linear(4, 1))

