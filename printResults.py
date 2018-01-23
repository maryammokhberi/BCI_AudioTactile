# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 21:59:01 2017

@author: Amarantine
"""

path="E:\Biomedical.master\Data\par011_2" 
os.chdir(path)



import pickle
with open ('analysisResults_10fold10repeat', 'rb') as fp:
    analysisResults_10fold10repeat = pickle.load(fp)
print " AT roc_auc_mean is", np.mean(analysisResults_10fold10repeat['roc_auc'])
print "AT roc_auc_std is", np.std(analysisResults_10fold10repeat['roc_auc'])
print " AT accuracy_mean is", np.mean(analysisResults_10fold10repeat['accuracy'])
print "AT accuracy_std is", np.std(analysisResults_10fold10repeat['accuracy'])

analysisResults_5fold=np.load('analysisResults.npy')

paired_ttest=np.load('paired_ttest.npy')
print "p-value", paired_ttest

Shapiro=np.load('Shapiro_NoMed.npy')
print "Shapiro", Shapirol

wilcoxon=np.load('wilcoxon_sign_rank.npy')
print "wilcoxon", wilcoxon

    with open ('meditationRest_analysisResults_5fold', 'rb') as fp:
        meditationRest_analysisResults_5fold = pickle.load(fp)
        
    print "meditation roc_auc_mean" , np.mean(meditationRest_analysisResults_5fold['roc_auc'])
    print "meditation roc_Auc_std", np.std(meditationRest_analysisResults_5fold['roc_auc'])         


#%%
MeditationSessions=['par011_1','par020_2','par040_2','par050_2','par061_1',
                   'par071_1','par080_2','par090_2','par101_1','par111_1',
                   'par120_2','par130_2']


NoMeditationSessions=['par011_2','par020_1','par040_1','par050_1','par061_2',
                   'par071_2','par080_1','par090_1','par101_2','par111_2',
                   'par120_1','par130_1']

for participant, location in enumerate(MeditationSessions):
    participant=participant
    path='E:\Biomedical.master\Data\\' + location
    os.chdir(path)
    print path
    with open ('analysisResults_10fold10repeat', 'rb') as fp:
        analysisResults_10fold10repeat = pickle.load(fp)
    print"accuracy_mean", np.mean(analysisResults_10fold10repeat['accuracy'])
    print"accuracy_std", np.std(analysisResults_10fold10repeat['accuracy'])
        
        