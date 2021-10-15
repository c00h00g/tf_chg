# -*- coding:utf-8 -*-

"""
将可迭代对象打包成元组
"""

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = [9, 10]

zip_ab = zip(a, b)
zip_ac = zip(a, c)

#(1, 5)
#(2, 6)
#(3, 7)
#(4, 8)
for elem in zip_ab:
    print(elem)

#与最短的保持一致
#(1, 9)
#(2, 10)
for elem in zip_ac:
    print(elem)

# 元素首尾相连
# [(1, 2), (2, 3), (3, 4)]
print zip(a[:-1], a[1:])
