# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:06:08 2017

@author: mokhberim
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm

x=x_imbalanced
y=y_imbalanced

C=1
#
#AT_svc=svm.SVC(kernel='linear', C=C, probability=True).fit(x, y)
#
#AT_rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True).fit(x,y)
#
#AT_poly_svc = svm.SVC(kernel='poly', degree=3, C=C, probability=True).fit(x,y)
#
#AT_lin_svc = svm.LinearSVC(C=C).fit(x,y)
#
#from sklearn.model_selection import cross_val_score
#roc_auc_score=cross_val_score(AT_svc, x, y, cv=5, scoring='roc_auc')
#accuracy_score=cross_val_score(AT_svc, x, y, cv=5, scoring='accuracy')
#precision_score=cross_val_score(AT_svc, x, y, cv=5, scoring='precision')
#recall_score=cross_val_score(AT_svc, x, y, cv=5, scoring='recall')
#
#imbalanceDataMetrics={'roc_auc_score':roc_auc_score.mean(),\
#                    'accuracy_score':accuracy_score.mean(),\
#                    'precision_score':precision_score.mean(),\
#                    'recall_score':recall_score.mean() }


#from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(x)
print(kf) 
accuracy=list()
precision=list()
recall=list()
roc_auc=list()
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    AT_svc = svm.SVC(kernel='linear', C=C, probability=True).fit(x_train, y_train)
    y_pred = AT_svc.predict
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    roc_auc.append(roc_auc_score(y_test, y_pred))

print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)
