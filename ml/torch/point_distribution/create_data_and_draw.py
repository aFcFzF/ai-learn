'''
@file draw.py
@author afcfzf(9301462@qq.com)
'''

import numpy as np
import matplotlib.pyplot as plt

def tag_entry(x, y):
    return 0 if x ** 2 + y ** 2 < 1 else 1

def create_data(num_of_data: int):
    entry_list = []
    for i in range(num_of_data):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry_list.append([x, y, tag])
    return np.array(entry_list)

# data = np.array([
#     [-1.5700402 , -0.71957951],
#     [ 1.73087583, -0.82547038],
#     [ 0.74275162,  2.14099108],
#     [ 1.36449517,  0.64425576],
#     [-2.26159213, -0.26920494],
#     [-2.07729085,  0.00815841],
#     [ 2.30664987, -1.42956103],
#     [ 0.76063793, -0.88875996],
#     [ 0.19691568, -0.62025159],
#     [ 0.69418857, -0.10614533],
#     [-0.17856137, -0.75573421],
#     [-1.55907069,  0.71972188],
#     [-1.66565161,  0.09212829],
#     [ 0.20354882,  0.22380984],
#     [-0.60671789,  0.27877269],
#     [ 0.44634232,  0.54996998],
#     [-0.6685074 , -0.02047394],
#     [-1.95284919, -0.47142535],
#     [-0.25979665,  2.07020467],
#     [ 0.96765445,  0.7602378 ]
# ])
def drawImage(data, title):
    color = []
    for i in data[:, 2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")

    plt.scatter(data[:, 0], data[:, 1], color=color)
    plt.title(title)
    plt.show()

NUM_OF_DATA = 20

if __name__ == "__main__":
    data = create_data(NUM_OF_DATA)
    print(data)
    drawImage(data, 'demo')