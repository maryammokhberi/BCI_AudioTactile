# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:39:08 2017

@author: Amarantine
"""
#
#import numpy as np
#import matplotlib.pyplot as plt
#
#from sklearn.ensemble import RandomForestClassifier
#
#AT_randomForest=RandomForestClassifier().fit(x_imbalanced, y_imbalanced)
#
#from sklearn.model_selection import cross_val_score
#roc_auc_score=cross_val_score(AT_randomForest, x_balanced, y_balanced, cv=5, scoring='roc_auc')
#accuracy_score=cross_val_score(AT_randomForest, x_balanced, y_balanced, cv=5, scoring='accuracy')
#precision_score=cross_val_score(AT_randomForest, x_balanced, y_balanced, cv=5, scoring='precision')
#recall_score=cross_val_score(AT_randomForest, x_balanced, y_balanced, cv=5, scoring='recall')
#
#imbalanceDataMetrics={'roc_auc_score':roc_auc_score.mean(),\
#                    'accuracy_score':accuracy_score.mean(),\
#                    'precision_score':precision_score.mean(),\
#                    'recall_score':recall_score.mean() }

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

x=X_balanced
y=y_balanced

#removing the nan elements due to rejecting bad epochs
nan_elements=np.argwhere(np.isnan(x)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
x=np.delete(x, nan_index, axis=0)
nan_elements=np.argwhere(np.isnan(y)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
y=np.delete(y, nan_index, axis=0)

#scaling features between -1 and 1
x=preprocessing.minmax_scale(x,feature_range=(-1,1))

C=1


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

    AT_randomForest=RandomForestClassifier().fit(x_train, y_train)
    y_pred = AT_randomForest.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    roc_auc.append(roc_auc_score(y_test, y_pred))

print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)
