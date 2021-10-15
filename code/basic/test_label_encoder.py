# -*- coding:utf-8 -*-


import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(["a", "b", "c"])

res1 = le.transform(["a", "b"])
# [0 1]
print(res1)

res2 = le.transform(["c"])
# [2]
print(res2)
