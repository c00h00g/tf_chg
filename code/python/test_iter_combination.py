# -*- coding:utf-8 -*-

"""
所有的组合数据
"""

import itertools

for i, j in itertools.combinations(list(range(5)), 2):
    print("i:%s, j:%s" % (i,j ))

"""
i:0, j:1
i:0, j:2
i:0, j:3
i:0, j:4
i:1, j:2
i:1, j:3
i:1, j:4
i:2, j:3
i:2, j:4
i:3, j:4
"""
