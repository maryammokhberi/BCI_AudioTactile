# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 16:07:08 2017

@author: Amarantine
"""
import numpy as np
import os
import pyvttbl as pt
#df = pt.DataFrame()
import scipy

pathMed='E:\\Biomedical.master\\Data\\par120_2'
pathNoMed='E:\\Biomedical.master\\Data\\par120_1'

NoMed_results=np.load(os.path.join(pathNoMed,'analysisResults_10fold10repeat'))
Med_results=np.load(os.path.join(pathMed,'analysisResults_10fold10repeat'))

Shapiro_NoMed=scipy.stats.shapiro(NoMed_results['roc_auc'])
Shapiro_Med=scipy.stats.shapiro(Med_results['roc_auc'])

paired_ttest = scipy.stats.ttest_rel(NoMed_results['roc_auc'], Med_results['roc_auc'])
print "The t-statistic is %.3f and the p-value is %15.15f" % paired_ttest

wilcoxon_sign_rank = scipy.stats.wilcoxon( Med_results['roc_auc'],NoMed_results['roc_auc'])
print "The wilcoxon is %.3f and the p-value is %.9f." % wilcoxon_sign_rank

np.save(os.path.join(pathNoMed,'paired_ttest.npy'),paired_ttest)
np.save(os.path.join(pathNoMed,'wilcoxon_sign_rank.npy'),wilcoxon_sign_rank)
np.save(os.path.join(pathMed,'paired_ttest.npy'),paired_ttest)
np.save(os.path.join(pathNoMed,'Shapiro_NoMed.npy'),Shapiro_NoMed)
np.save(os.path.join(pathMed,'Shapiro_Med.npy'),Shapiro_Med)


#
#
##load resluts
#
##participant sessions
#NoMeditSessions=np.array(['par011_2','par020_1','par040_1','par050_1','par061_2',
#                 'par101_2','par111_2','par130_1'])
#MeditSessions=['par011_1','par020_2','par040_2','par050_2','par061_1',
#                 'par101_1','par111_1','par130_2']
##NoMedResultssvm2=np.full([8],np.nan)
#
#NoMedResults_conf_matrices=[None]*8
#NoMedResults_auc_roc_max=[None]*8
#NoMedResults_bestclf=[None]*8
#
#MedResults_conf_matrices=[None]*8
#MedResults_auc_roc_max=[None]*8
#MedResults_bestclf=[None]*8
#
##
##for i in range(NoMeditSessions.shape[0]):
##    path = 'E:\\Biomedical.master\\Data\\' + NoMeditSessions[i]
##    NoMedResultsi=np.load(os.path.join(path, 'analysisResults.npy'))
#    
##0:max_roc_au
##1:clf
##2:conf matrix?
#i=7
#path = 'E:\\Biomedical.master\\Data\\' + NoMeditSessions[i]
#NoMedResultsi=np.load(os.path.join(path, 'analysisResults.npy'))
#NoMedResults_bestclf[i]='svm2'
#NoMedResults_auc_roc_max[i]=NoMedResultsi.item().get(NoMedResults_bestclf[i])['roc_auc_mean']
#NoMedResults_conf_matrices[i]=NoMedResultsi.item().get(NoMedResults_bestclf[i])['confusion_matrix']
#
#
#i=7
#path = 'E:\\Biomedical.master\\Data\\' + MeditSessions[i]
#MedResultsi=np.load(os.path.join(path, 'analysisResults.npy'))
#MedResults_bestclf[i]='svm2'
#MedResults_auc_roc_max[i]=MedResultsi.item().get(MedResults_bestclf[i])['roc_auc_mean']
#MedResults_conf_matrices[i]=MedResultsi.item().get(MedResults_bestclf[i])['confusion_matrix']
#MedResults=[MedResults_bestclf,MedResults_auc_roc_max,MedResults_conf_matrices]
#
#
#import pickle
#with open('MedResults', 'wb') as fp:
#    pickle.dump(MedResults, fp)
#    
#with open('NoMedResults', 'wb') as fp:
#    pickle.dump(NoMedResults, fp)
#
#
##to read the pickle file back    
#with open ('MedResults', 'rb') as fp:
#    MedResults = pickle.load(fp)    
#    
#with open ('NoMedResults', 'rb') as fp:
#    NoMedResults = pickle.load(fp)    
##
##paired_sample = scipy.stats.ttest_rel(NoMedResults_auc_roc_max, MedResults_auc_roc_max)
##print "The t-statistic is %.3f and the p-value is %.3f." % paired_sample
##
##two_sample = scipy.stats.ttest_ind(NoMedResults_auc_roc_max, MedResults_auc_roc_max)
##print "The t-statistic is %.3f and the p-value is %.3f." % two_sample
##
#wilcoxon_sign_rank = scipy.stats.wilcoxon(NoMedResults_auc_roc_max, MedResults_auc_roc_max)
#print "The t-statistic is %.3f and the p-value is %.3f." % wilcoxon_sign_rank
#
#
#with open('testResults', 'wb') as fp:
#    pickle.dump(testResults, fp)
#
#
#
#MedResults_confmat_1stfold_precision=[None]*8
#MedResults_confmat_1stfold_recall=[None]*8
#MedResults_confmat_2ndfold_precision=[None]*8
#MedResults_confmat_2ndfold_recall=[None]*8
#for i, confMat_participants in enumerate(MedResults_conf_matrices):
#    MedResults_confmat_1stfold_precision[i]=confMat_participants[0][1,1]/float(confMat_participants[0][1,1]+confMat_participants[0][1,0])
#    MedResults_confmat_1stfold_recall[i]=confMat_participants[0][1,1]/float(confMat_participants[0][1,1]+confMat_participants[0][0,1])
#    MedResults_confmat_2ndfold_precision[i]=confMat_participants[1][1,1]/float(confMat_participants[1][1,1]+confMat_participants[1][1,0])
#    MedResults_confmat_2ndfold_recall[i]=confMat_participants[1][1,1]/float(confMat_participants[1][1,1]+confMat_participants[1][0,1])
#
#NoMedResults_confmat_1stfold_precision=[None]*8
#NoMedResults_confmat_1stfold_recall=[None]*8
#NoMedResults_confmat_2ndfold_precision=[None]*8
#NoMedResults_confmat_2ndfold_recall=[None]*8
#for i, confMat_participants in enumerate(NoMedResults_conf_matrices):
#    NoMedResults_confmat_1stfold_precision[i]=confMat_participants[0][1,1]/float(confMat_participants[0][1,1]+confMat_participants[0][1,0])
#    NoMedResults_confmat_1stfold_recall[i]=confMat_participants[0][1,1]/float(confMat_participants[0][1,1]+confMat_participants[0][0,1])
#    NoMedResults_confmat_2ndfold_precision[i]=confMat_participants[1][1,1]/float(confMat_participants[1][1,1]+confMat_participants[1][1,0])
#    NoMedResults_confmat_2ndfold_recall[i]=confMat_participants[1][1,1]/float(confMat_participants[1][1,1]+confMat_participants[1][0,1])

