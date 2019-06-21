# -*- coding:utf-8 -*-

import sys
import numpy as np

a = [1.1, 4.5, 5.6]

#最近的偶数
#For values exactly halfway between rounded decimal values, NumPy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
print np.round(a)

#向上取整
print np.ceil(a)

#向下取整
print np.floor(a)
