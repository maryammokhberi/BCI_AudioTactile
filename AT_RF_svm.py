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
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


data = np.concatenate([X, y], axis = 1)

#removing the nan elements due to rejecting bad epochs
nan_elements=np.argwhere(np.isnan(data)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
data =np.delete(data, nan_index, axis=0)

data_x=data[:,0:-1]        
data_y=data[:,-1]

#scaling features between -1 and 1
data_x=preprocessing.minmax_scale(data_x,feature_range=(-1,1))
#x=preprocessing.scale(x, axis = 0 )


#from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(data_x)
print(kf) 
accuracy=list()
precision=list()
recall=list()
roc_auc=list()
conf = np.zeros((2,2))
for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data_x[train_index], data_x[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]
    
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
    
    
    X_train_balanced = x_train      
    y_train_balanced = y_train
    
    # Random Forests Classifier
    AT_randomForest=RandomForestClassifier(n_estimators=100, 
                                           class_weight={0:1, 1:7})
    AT_randomForest.fit(X_train_balanced, y_train_balanced)
    y_prob_pred=AT_randomForest.predict_proba(x_test)
    y_pred = AT_randomForest.predict(x_test)
    
#    
#    # SVM classifier
#    AT_svc = svm.SVC(kernel='linear', 
#                     C=1, probability=True, class_weight='balanced')
#    AT_svc.fit(x_train, y_train)
#    y_prob_pred=AT_svc.predict_proba(x_test)
#    y_pred = AT_svc.predict(x_test)
    
    
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    roc_auc.append(roc_auc_score(y_test, y_pred))
    conf += confusion_matrix(y_test, y_pred)
    
precision_curve, recall_curve, _ = precision_recall_curve(
        y_test, y_prob_pred[:,0],  sample_weight = [7 if i == 1 else 1 for i in y_train])
fpr, tpr, thresholds = roc_curve(y_test, y_prob_pred[:,0], pos_label=1)
plt.figure(1)
plt.plot(fpr,tpr)
plt.figure(2)
plt.plot(precision_curve, recall_curve)
plt.figure(3)
plt.imshow(confusion_matrix(y_test, y_pred))

print np.max(accuracy)
print np.max(precision)
print np.max(recall)
print np.max(roc_auc)
