'''
@file split.py
@author afcfzf(9301462@qq.com)
'''

import torch
from transformers import BertTokenzier

# 示例文本
text = """Python是一种流行的编程语言。它以简洁的语法和丰富的生态系统而闻名。
Python支持多种编程范式。包括面向对象编程、函数式编程等。
它的应用领域非常广泛。从Web开发到人工智能，Python都有广泛应用。"""

def split_by_length(text, chunk_size = 50):
    result = []
    for i in range(0, len(text), chunk_size):
        result.append(text[i:i +chunk_size])
        # print("start: " + str(i) + " end: " + str(i + chunk_size))
        # print(text[i:i +chunk_size])
    return result

chunks = split_by_length(text)

# print("分割后的文本块数量: ", len(chunks), chunks)

for idx, chunk in enumerate(chunks):
    print(f"\n块 {idx}:")
    print(chunk)
