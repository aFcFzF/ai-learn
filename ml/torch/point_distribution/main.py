'''
@file main.py
@author afcfzf(9301462@qq.com)
'''

import numpy as np
import numbers as num
import matplotlib.pyplot as plt
import create_data_and_draw as cd

# draw.drawImage()

NET_WORK_SHAPE = [2, 3, 4, 5, 2]

BATCH_SIZE = 5

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
    
    def normal(self, np_inputs):
        row_max = np.max(np.absolute(np_inputs), axis=1, keepdims=True)
        scale_rate = np.where(row_max == 0, 0, 1 / row_max)
        return np_inputs * scale_rate
    
    def get_weight_adjust_matrix(self, pre_weight_vals, after_weight_demands):
        # 
        plain_weights = np.full(self.weights.shape, 1)
        # 权重调整矩阵
        weights_adjust_matrix = np.full(self.weights.shape, 0)
        # 转置矩阵
        plain_weights_T = plain_weights.T
        
        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T * pre_weight_vals[i, :]).T * after_weight_demands[i, :]
         
        

# print(Layer(2, 3).normal([[0.9, -0.4], [-0.8, 0.5], [-0.5, 0.8]]))
# print(Layer(2, 3).layer_forward([[1, 2], [3, 4]]))
# print(Layer(2, 3).activate_softmax([[0.9, -0.4], [-0.8, 0.5], [-0.5, 0.8]]))

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
    def network_forward(self, inputs) -> list[np.ndarray]:
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
            # print('=== res:')
            # print(res)
            # print('\n')
        
        return outputs
    
    def classify(self, prob):
        return np.rint(prob[:, 1])
    
    def convert_result_matrix(self, val: list[float]):
        matrix = np.zeros((len(val), 2))
        matrix[:, 1] = val
        matrix[:, 0] = 1 - val
        return matrix
    
    def precise_loss_function(self, predicted: np.ndarray, real: np.ndarray):
        print("real_matrix: \n", real)
        print("predicted \n", predicted)
        product = np.sum(predicted * real, axis=1)
        return 1 - product
    
    # 需求函数: 只有最后一层需要
    def get_final_layer_predict_demands(self, predicted: np.ndarray, real: np.ndarray):
        target = np.zeros((len(predicted), 2))
        for i in range(len(predicted)):
            if np.dot(predicted[i], real[i]) > .5:
                target[i] = np.array([0, 0])
            else:
                target[i] = (real[i] - .5) * 2
        return target

# n = Network().network_forward([[1, 2], [3, 4]])
def main():
    data = cd.create_data(5)
    real = data[:, 2]
    print("== data \n", data)
    print("== real \n", real)

    cd.drawImage(data, "real")

    inputs = data[:, (0, 1)]
    n = Network(NET_WORK_SHAPE)
    output = n.network_forward(inputs)
    predict = output[-1]
    real_mrx = n.convert_result_matrix(real)
    loss = n.precise_loss_function(predict, real_mrx)
    print("loss \n", loss)
    demand = n.get_final_layer_predict_demands(predict, real_mrx)
    print("demand \n", demand)

    # 分类
    data[:, 2] = n.classify(predict)
    print("classify\n", data)
    cd.drawImage(data, "before train")

if __name__ == '__main__':
    main();