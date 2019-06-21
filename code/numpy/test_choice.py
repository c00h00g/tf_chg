# -*- coding:utf-8 -*-

"""
选择随机数
numpy.random.choice(a, size=None, replace=True, p=None)
a : If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)
size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
replace : Whether the sample is with or without replacement
"""

import numpy as np

a = [2, 4, 6, 8 ,10, 12, 14, 16]

# [ 2 14  2 16 10]
# 可以有重复
b = np.random.choice(a, 5)

#[14 12  4  2 16]
# 不可以有重复
c = np.random.choice(a, 5, replace = False)

print(b)
print(c)
