# -*- coding:utf-8 -*-
# 靠靠靠靠靠靠靠靠

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 靠
print(np.vstack((a, b)))

# 靠 Horizontal
print(np.hstack((a, b)))

"""output
[[1 2 3]
 [4 5 6]]

[1 2 3 4 5 6]
"""
