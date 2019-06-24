# -*- coding:utf-8 -*-
# 排列

import numpy as np

a = [ x for x in range(10)]

#[3 2 1 7 5 6 0 9 4 8]
b = np.random.permutation(a)

# 根据长度
# [2 1 4 3 0]
c = np.random.permutation(5)
print(c)

print(a)
print(b)
#numpy.ndarray
print(type(b))

