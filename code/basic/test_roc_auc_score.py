# -*- coding:utf-8 -*-



from sklearn.metrics import roc_auc_score
import numpy as np

y_true = np.array([0, 0, 0, 1])
y_pred = np.array([0.1, 0.1, 0.2, 0.3])


auc = roc_auc_score(y_true, y_pred)
print(auc)
