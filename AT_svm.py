# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:06:08 2017

@author: mokhberim
"""

#def AT_svm(X,y):
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
np.random.seed(123)




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
kf.get_n_splits(X)
print(kf) 
accuracy=list()
precision=list()
recall=list()
roc_auc=list()
conf = np.zeros((2,2))
evidence_accumulation_score_all=np.full([5], np.nan)
fold=-1
for train_index, test_index in kf.split(X):
    fold+=1
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
#    
##oversampling the oddball stimuli to fix imabalance of the data
#    Xy=np.concatenate([X_train,y_train[:,np.newaxis]],axis=1)
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
#    X_train=Xy_balanced[:,0:-1]        
#    y_train=Xy_balanced[:,-1]
#    
    
    
    
    data_Xy_train = np.concatenate([X_train, y_train], axis = 1)

    #removing the nan elements due to rejecting bad epochs
    nan_elements=np.argwhere(np.isnan(data_Xy_train)==True)
    nan_index=nan_elements[:,0]
    nan_index=np.unique(nan_index)
    data_Xy_train =np.delete(data_Xy_train, nan_index, axis=0)
    
    data_X_train=data_Xy_train[:,0:-1]        
    data_y_train=data_Xy_train[:,-1]
    #scaling features between -1 and 1
    data_X_train=preprocessing.minmax_scale(data_X_train,feature_range=(-1,1))
    #x=preprocessing.scale(X, axis = 0 )
    
    
    #training the classifier
    AT_svc = svm.SVC(kernel='linear', C=1, probability=True,class_weight='balanced').fit(data_X_train, data_y_train)
    
    
    #testing the classifier

        
    data_Xy_test = np.concatenate([X_test, y_test], axis = 1)
    #removing the nan elements 
    nan_elements=np.argwhere(np.isnan(data_Xy_test)==True)
    nan_index=nan_elements[:,0]
    nan_index=np.unique(nan_index)
    data_Xy_test =np.delete(data_Xy_test, nan_index, axis=0)
    data_X_test=data_Xy_test[:,0:-1]        
    data_y_test=data_Xy_test[:,-1]
    
    #scaling features between -1 and 1
    data_X_test=preprocessing.minmax_scale(data_X_test,feature_range=(-1,1))
    #x=preprocessing.scale(X, axis = 0 )
           
    
#            data_y_pred=AT_svc.predict(data_X_test)
    data_y_pred_prob=AT_svc.predict_proba(data_X_test)
#            print data_y_pred_prob
    data_y_pred=[1 if i<.8 else 0 for i in data_y_pred_prob[:,0]]
    data_y_pred=np.asarray(data_y_pred)

    accuracy.append(accuracy_score(data_y_test, data_y_pred))
    precision.append(precision_score(data_y_test, data_y_pred))
    recall.append(recall_score(data_y_test, data_y_pred))
    roc_auc.append(roc_auc_score(data_y_test, data_y_pred))
    conf += confusion_matrix(data_y_test, data_y_pred)
    
precision_curve, recall_curve, _ = precision_recall_curve(
        data_y_test, data_y_pred_prob[:,0],  sample_weight = [7 if i == 1 else 1 for i in y_train])
fpr, tpr, thresholds = roc_curve(data_y_test, data_y_pred_prob[:,0], pos_label=1)
plt.figure(1)
plt.plot(fpr,tpr)
plt.figure(2)
plt.plot(precision_curve, recall_curve)
plt.figure(3)
#plt.imshow(confusion_matrix(y_test, data_y_pred))
print conf
print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)

            

            

    
    
    
    
    
    
    

    
        

#accuracy.append(accuracy_score(y_test, y_pred))
#precision.append(precision_score(y_test, y_pred))
#recall.append(recall_score(y_test, y_pred))
#roc_auc.append(roc_auc_score(y_test, y_pred))
#conf += confusion_matrix(y_test, y_pred)
#
#precision_curve, recall_curve, _ = precision_recall_curve(
#        y_test, y_prob_pred[:,0],  sample_weight = [7 if i == 1 else 1 for i in y_train])
#fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred[:,0], pos_label=1)
##plt.figure(1)
##plt.plot(fpr,tpr)
##plt.figure(2)
##plt.plot(precision_curve, recall_curve)
##plt.figure(3)
##plt.imshow(confusion_matrix(y_test, y_pred))
#print conf
#print np.max(accuracy)
#print np.max(precision)
#print np.max(recall)
#print np.max(roc_auc)
