'''
@file draw.py
@author afcfzf(9301462@qq.com)
'''

import numpy as np
import matplotlib.pyplot as plt

def drawImage():
    data = np.array([
        [-1.5700402 , -0.71957951],
        [ 1.73087583, -0.82547038],
        [ 0.74275162,  2.14099108],
        [ 1.36449517,  0.64425576],
        [-2.26159213, -0.26920494],
        [-2.07729085,  0.00815841],
        [ 2.30664987, -1.42956103],
        [ 0.76063793, -0.88875996],
        [ 0.19691568, -0.62025159],
        [ 0.69418857, -0.10614533],
        [-0.17856137, -0.75573421],
        [-1.55907069,  0.71972188],
        [-1.66565161,  0.09212829],
        [ 0.20354882,  0.22380984],
        [-0.60671789,  0.27877269],
        [ 0.44634232,  0.54996998],
        [-0.6685074 , -0.02047394],
        [-1.95284919, -0.47142535],
        [-0.25979665,  2.07020467],
        [ 0.96765445,  0.7602378 ]
    ])

    x = data[:, 0]
    y = data[:, 1]

    colors = []
    for idx in range(len(x)):
        if x[idx] ** 2 + y[idx] ** 2 > 1:
            colors.append([0, 0, 1])
        else:
            colors.append([1, 0, 0])

    plt.scatter(x=x, y=y, color=colors)
    plt.show()