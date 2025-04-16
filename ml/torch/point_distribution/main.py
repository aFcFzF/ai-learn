'''
@file main.py
@author afcfzf(9301462@qq.com)
'''

import numpy as np
import numbers as num
import matplotlib.pyplot as plt
# import draw

# draw.drawImage()

def printTable(data):
    table = []
    for row in data:
        formatted_row = [f"{x:8.3f}" for x in row]  # 固定宽度8字符，1位小数
        table.append("| " + " | ".join(formatted_row) + " |")

    # 添加分隔线
    separator = "-" * (len(table[0]) - 2)  # 根据第一行长度生成分隔线
    print("\n".join([separator] + table + [separator]))

class Layer:
    def activate_relu(self, num):
        return np.maximum(0, num)
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
    
    # 归一化，入参 [[4, 3], [0, 0]] → [[0.57, 0.42], [0, 0]]
    # 简单的row[i] / (sum(row))处理，可以举个反例吗？[-1, 1] → [nan, nan]
    # 使用e ** row[i] - max(row)的指数函数, 最大值是0, 剩下的都是负数，指数函数介于[0, 1]： row[i] - max(row) = [-2, 0]
    def activate_softmax(self, np_inputs):
        row_max = np.max(np_inputs, axis=1, keepdims=True)
        res = np.exp(np_inputs - row_max)
        return res / np.sum(res, axis=1, keepdims=True)

        # arr = np.absolute(np.array(np_inputs))
        # return arr / arr.sum(axis=1, keepdims=True)
    def normal(self, np_inputs):
        row_max = np.max(np.absolute(np_inputs), axis=1, keepdims=True)
        scale_rate = np.where(row_max == 0, 0, 1 / row_max)
        return np_inputs * scale_rate

# Layer(2, 3).normal([[-9, 1], [-0.23, 10], [1, 2]])
# print(Layer(2, 3).layer_forward([[1, 2], [3, 4]]))
# print(Layer(2, 3).activate_softmax([[-1, 1], [1, 2], [3, 4]]))

# [[1, 2]] → [[1, 2, 3]]

class Network: # 网络形状
    shape: list[int]

    # 网络层
    layers: list[Layer] = []

    def __init__(self, network_shape = [2, 3, 4, 2]):
        self.shape = network_shape
        for i in range(len(network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i + 1])
            self.layers.append(layer)

    # 前馈运算：转成最终输出的概率，inputs就是输入值
    def network_forward(self, inputs) -> np.ndarray:
        outputs = [inputs]
        length = len(self.layers)

        for idx in range(length):
            layer = self.layers[idx]
            forward = layer.layer_forward(outputs[idx])
            if idx < length - 1:
                res = layer.activate_relu(forward)
                res = layer.normal(res)
                outputs.append(res)
            else:
                res = layer.activate_softmax(forward)
                outputs.append(res)

            # printTable(res)
            print('=== res:')
            print(res)
            print('\n')
        
        return outputs

n = Network().network_forward([[1, 2], [3, 4]])
