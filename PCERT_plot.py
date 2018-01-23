import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy

questionnaire_data=dict(S1_meditation=[2,2,3,1,2],
                        S1_bci_control=[8,8,5,7,5],
                        S1_bci_study=[4,5,4,6,2],
                        S2_meditation=[6,1,4,1,4],
                        S2_bci_control=[2,5,4,7,0],
                        S2_bci_study=[0,4,4,7,0],
                        S3_meditation=[2,3,1,1,1],
                        S3_bci_control=[1,6,1,0,0],
                        S3_bci_study=[1,2,0,0,0],
                        S4_meditation=[2,1,2,4,1],
                        S4_bci_control=[2,8,5,6,2],
                        S4_bci_study=[5,8,4,5,2],
                        S5_meditation=[3,1,3,1,1],
                        S5_bci_control=[1,4,5,7,0],
                        S5_bci_study=[6,7,1,5,0],
                        S6_meditation=[7,5,3,2,1],
                        S6_bci_control=[5,6,1,2,0],
                        S6_bci_study=[4,4,0,2,0],
                        S7_meditation=[2,1,3,1,2],
                        S7_bci_control=[4,6,3,5,3],
                        S7_bci_study=[2,4,3,3,3],
                        S8_meditation=[2,1,3,1,1],
                        S8_bci_control=[7,5,2,1,2],
                        S8_bci_study=[8,5,5,5,1],
                        S9_meditation=[1,2,1,2,4],
                        S9_bci_control=[1,2,5,6,5],
                        S9_bci_study=[2,5,5,6,9],
                        S10_meditation=[2,3,2,1,2],
                        S10_bci_control=[4,7,1,7,1],
                        S10_bci_study=[5,8,2,7,0],
                        S11_meditation=[4,6,4,7,1],
                        S11_bci_control=[3,5,3,6,1],
                        S11_bci_study=[7,8,1,6,0],
                        S12_meditation=[2,2,2,1,2],
                        S12_bci_control=[3,4,1,0,2],
                        S12_bci_study=[4,5,2,3,0])

#with open('questionnaire_data', 'wb') as fp:
#    pickle.dump(questionnaire_data, fp)


#%%
PCERT_level_meditation_mean=np.full((12),np.nan)
PCERT_level_bci_control_mean=np.full((12),np.nan)
PCERT_level_bci_study_mean=np.full((12),np.nan)

for i in range(12) :
        PCERT_level_meditation_mean[i]=np.mean(questionnaire_data['S'+str(i+1)+'_meditation'][2])
        PCERT_level_bci_control_mean[i]=np.mean(questionnaire_data['S'+str(i+1)+'_bci_control'][0])
        PCERT_level_bci_study_mean[i]=np.mean(questionnaire_data['S'+str(i+1)+'_bci_study'][0])





meditation_vs_rest=[81,78,99,98,100,80,67,64,93,93,63,89]
bci_performance_control=[61,86,61,79,76,61,64,54,70,72,51,90]
bci_performance_study=[60,71,68,75,77,52,58,68,80,73,70,97]
age=[17 ,17 ,16 ,17 ,16 ,17 ,10 ,11 ,14 ,15 ,16 ,17]

#%%
print scipy.stats.pearsonr(PCERT_level_meditation_mean,meditation_vs_rest)
print scipy.stats.pearsonr(PCERT_level_bci_control_mean,bci_performance_control)
print scipy.stats.pearsonr(PCERT_level_bci_study_mean,bci_performance_study)
print scipy.stats.pearsonr(age,meditation_vs_rest)
print scipy.stats.pearsonr(age,bci_performance_control)
print scipy.stats.pearsonr(age,bci_performance_study)

#%%
#demographic analysis between 
bci_performance_control_female=[61,86,76,61,54,72,51,90]
bci_performance_control_male=[61,79,64,70]
bci_performance_study_female=[60,71,77,52,68,73,70,97]
bci_performance_study_male=[68,75,58,80,]
meditation_vs_rest_female=[81,78,100,80,64,93,63,89]
meditation_vs_rest_male=[99,98,67,93]

print scipy.stats.mannwhitneyu(bci_performance_control_female,bci_performance_control_male)
print scipy.stats.mannwhitneyu(bci_performance_study_female,bci_performance_study_male)
print scipy.stats.mannwhitneyu(meditation_vs_rest_female,meditation_vs_rest_male)


