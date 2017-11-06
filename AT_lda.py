# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 15:29:36 2017

@author: mokhberim
"""

#there is 5 steps in lda
#1: compute d-dimensional mean vectors for different calsses: (here 2 matrices of 1*n-feature matrix)
#2: compute scatter matrices (both between class and within class)
#3: compute eigen vectors and corresponding eigen values
#4: sort eigen vectors by decreasing eigen vallues and choose k eigen vectors 
#...: with highest eigen values to form a   d*k matrice of W
#5: use the d*k eigen matrix to transform samples to a new subspace. 
#...: this can be summerized by Z=X*W (where X is n*d matrix (n-samples*n-features)
#...:, W is the d*k eigen vectors and Z is the tramsformed samples into the new subspace)

#import numpy as np
#import matplotlib.pyplot as plt
#
#
#def AT_lda_train (x,y):
#    
##import matplotlib.pyplot as plt
#    from sklearn import lda
#    
#    clf_lda = lda.LDA()
#    clf_lda.fit(x, y)    
#    x_transformed=clf_lda.fit_transform(x,y)
#    plt.figure(figsize = (35, 20))
#    plt.scatter(np.arange(0,x_transformed.shape[0]), x_transformed[:, 0],s=100, marker='D')
#    return clf_lda
#    
#def AT_lda_crossVal (clf,x,y):
#    from sklearn.model_selection import cross_val_score
#    scores= cross_val_score(clf, x, y)
#    accuracy=scores.mean()
#    
#    return accuracy
#    
#    
#def AT_lda_classify(clf, x):
#    
#    
#    stim_Class=clf.predict(x)
#    print stim_Class
#    x_transformed=clf.transform(x)
#    plt.scatter(np.arange(0,x_transformed.shape[0]),x_transformed,s=100, marker='o')
#    return stim_Class

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x=X
y=y

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
    clf_lda = LDA()
    AT_lda=clf_lda.fit(x_train, y_train)
    y_pred = AT_lda.predict(x_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    roc_auc.append(roc_auc_score(y_test, y_pred))

print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)

