'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-06 10:31:27
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 14:22:30
FilePath: /resnetV2/utils/norm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np


def x_norm(x):
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        
    x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm
    return x