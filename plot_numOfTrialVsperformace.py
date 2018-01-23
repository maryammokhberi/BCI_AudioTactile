# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 10:19:27 2018

@author: Amarantine
"""
import numpy as np
import os 
import pickle
MeditationSessions=['par011_1','par020_2','par040_2','par050_2','par061_1',
                   'par071_1','par080_2','par090_2','par101_1','par111_1',
                   'par120_2','par130_2']


NoMeditationSessions=['par011_2','par020_1','par040_1','par050_1','par061_2',
                   'par071_2','par080_1','par090_1','par101_2','par111_2',
                   'par120_1','par130_1']

meta_roc_auc_mean=np.full((2,12,12),np.nan) # 2 sessions, 12 participatnts, 12 trials
for participant, location in enumerate(MeditationSessions):
    participant=participant
    path='E:\Biomedical.master\Data\\' + location
    os.chdir(path)
    print path
    with open ('analysisResults_NumOftrials', 'rb') as fp:
        analysisResults_NumOftrials = pickle.load(fp)
    meta_roc_auc_mean[1][participant][:]=analysisResults_NumOftrials['meta_roc_auc_mean']    
    
for participant, location in enumerate(NoMeditationSessions):
    participant=participant
    path='E:\Biomedical.master\Data\\' + location
    os.chdir(path)
    print path
    with open ('analysisResults_NumOftrials', 'rb') as fp:
        analysisResults_NumOftrials = pickle.load(fp)
    meta_roc_auc_mean[0][participant][:]=analysisResults_NumOftrials['meta_roc_auc_mean']  
#%%


import numpy as np
import matplotlib.pyplot as plt

plt.figure()

for i in range(12):
    trial = np.arange(1,13)
    
    y_meditation=meta_roc_auc_mean_forThesis[1][i]*100

    
    ## example data
    #x = np.arange(0.1, 4, 0.5)
    #y = np.exp(-x)
    
    # example variable error bar values
    #yerr = 0.1 + 0.2*np.sqrt(x)
    #xerr = 0.1 + yerr
    
    # First illustrate basic pyplot interface, using defaults where possible.
    textposition=np.argmax(y_meditation)
    plt.axis([0, 12,  0, 100])
    # Now switch to a more OO interface to exercise more features.
    color=np.random.rand(3,)
    plt.plot(trial, y_meditation    ,color=color)
    
    plt.errorbar(trial, y_meditation, yerr=0, fmt='o',color=color)
    plt.text((textposition+1)-0.3*(i%2),y_meditation[textposition] ,'S'+str(i+1) , fontsize=10,color=color)
    plt.title('classifiaction Performance Vs. Number of Trials for Different Participants')
    plt.xlabel('Number of trials')
    plt.ylabel('AUC')
    
    

    plt.show()
#%%
        
    
    
    