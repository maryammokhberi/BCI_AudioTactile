# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 15:38:59 2017

@author: Amarantine
"""


#def AT_svm(X,y):
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score
from sklearn import svm
import matplotlib.pyplot as plt


data = np.concatenate([X, y], axis = 1)

#removing the nan elements due to rejecting bad epochs
nan_elements=np.argwhere(np.isnan(data)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
data =np.delete(data, nan_index, axis=0)

data_X=data[:,0:-1]        
data_y=data[:,-1]

#scaling features between -1 and 1
data_X=preprocessing.minmax_scale(data_X,feature_range=(-1,1))
#x=preprocessing.scale(x, axis = 0 )



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




accuracy=list()
precision=list()
recall=list()
roc_auc=list()
conf = np.zeros((2,2))

    
         #oversampling the oddball stimuli to fix imabalance of the data
#    Xy=np.concatenate([x_train,y_train[:,np.newaxis]],axis=1)
#    j = 0
#    for i in range(Xy.shape[0]):
#          
#        if Xy[i,-1]==1:
#            temp=np.tile(Xy[i,:], (7,1)) # Xy_balanced[j:j+7,:]
#            j=j+7
#        else:
#            temp=Xy[i,:] #Xy_balanced[j,:]
#            temp = temp[np.newaxis, :]
#            j=j+1
#            
#        if i==0:
#            Xy_balanced = temp
#        else:
#            Xy_balanced = np.concatenate([Xy_balanced, temp], axis = 0)
#    
#    X_train_balanced=Xy_balanced[:,0:-1]        
#    y_train_balanced=Xy_balanced[:,-1]
    
        
    
    
    
AT_svc = svm.SVC(kernel='linear', C=C, probability=True, class_weight='balanced').fit(data_X, data_y)
#y_prob_pred=AT_svc.predict_proba(data_X)

#y_pred = AT_svc.predict(data_X)

    

accuracy.append(accuracy_score(data_y, y_pred))
precision.append(precision_score(data_y, y_pred))
recall.append(recall_score(data_y, y_pred))
roc_auc.append(roc_auc_score(data_y, y_pred))
conf += confusion_matrix(data_y, y_pred)

#precision_curve, recall_curve, _ = precision_recall_curve(
#        data_y, y_prob_pred[:,0],  sample_weight = [7 if i == 1 else 1 for i in data_y] )
#fpr, tpr, thresholds = roc_curve(data_y, y_pred, pos_label=1)
#plt.figure(1)
#plt.plot(fpr,tpr)
#plt.figure(2)
#plt.plot(precision_curve, recall_curve)

print conf
print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)

#return AT_svc