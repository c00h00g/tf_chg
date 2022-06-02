# -*- coding:utf-8 -*-

# 使用二进制的方式存储数组
"""
chg a is -------->
[[6 6 4 2 4]
 [6 7 7 8 3]
 [7 0 0 6 5]]
chg c is -------->
[[6 6 4 2 4]
 [6 7 7 8 3]
 [7 0 0 6 5]]
"""

import numpy as np

a = np.random.randint(0, 10, size=(3, 5))
print("chg a is -------->")
print(a)

b = np.save("data.npy", a)

c = np.load("data.npy")
print("chg c is -------->")
print(c)
