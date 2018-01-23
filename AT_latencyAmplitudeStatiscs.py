# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 22:11:53 2018

@author: Amarantine
"""
import pickle 
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt



MeditationSessions=['par011_1','par020_2','par040_2','par050_2','par061_1',
                   'par071_1','par080_2','par090_2','par101_1','par111_1',
                   'par120_2','par130_2']

NoMeditationSessions=['par011_2','par020_1','par040_1','par050_1','par061_2',
                   'par071_2','par080_1','par090_1','par101_2','par111_2',
                   'par120_1','par130_1']


#12 participants,2 session each, 4 quantity(latenacy mean0 and std1, amplitude mean2 and std3), 8 words
word_lat_amp_results_detail=np.full((12,2,4,8),np.nan)
speaker_lat_amp_results_detail=np.full((12,2,4,4),np.nan)
      

for participant, location in enumerate(NoMeditationSessions):
    participant=participant
    path='E:\Biomedical.master\Data\\' + location
    os.chdir(path)
    print path
    with open ('AT_latencyAmplitudeAnalysis', 'rb') as fp:
        AT_latencyAmplitudeAnalysis = pickle.load(fp)
    #no meditation session
    for word in range(8):
        word_lat_amp_results_detail[participant][0][0][word]=np.nanmean(AT_latencyAmplitudeAnalysis['latency_word'][word])
        word_lat_amp_results_detail[participant][0][1][word]=np.nanstd(AT_latencyAmplitudeAnalysis['latency_word'][word])
        word_lat_amp_results_detail[participant][0][2][word]=np.nanmean(AT_latencyAmplitudeAnalysis['amplitude_word'][word])
        word_lat_amp_results_detail[participant][0][3][word]=np.nanstd(AT_latencyAmplitudeAnalysis['amplitude_word'][word])
    for speaker in range(4):     
        speaker_lat_amp_results_detail[participant][0][0][speaker]=np.nanmean(AT_latencyAmplitudeAnalysis['latency_word'][speaker])
        speaker_lat_amp_results_detail[participant][0][1][speaker]=np.nanstd(AT_latencyAmplitudeAnalysis['latency_word'][speaker])
        speaker_lat_amp_results_detail[participant][0][2][speaker]=np.nanmean(AT_latencyAmplitudeAnalysis['amplitude_word'][speaker])
        speaker_lat_amp_results_detail[participant][0][3][speaker]=np.nanstd(AT_latencyAmplitudeAnalysis['amplitude_word'][speaker])

for participant, location in enumerate(MeditationSessions):
    participant=participant
    path='E:\Biomedical.master\Data\\' + location
    print path
    os.chdir(path)
    with open ('AT_latencyAmplitudeAnalysis', 'rb') as fp:
        AT_latencyAmplitudeAnalysis = pickle.load(fp)
    # meditation session
    for word in range(8):
        word_lat_amp_results_detail[participant][1][0][word]=np.nanmean(AT_latencyAmplitudeAnalysis['latency_word'][word])
        word_lat_amp_results_detail[participant][1][1][word]=np.nanstd(AT_latencyAmplitudeAnalysis['latency_word'][word])
        word_lat_amp_results_detail[participant][1][2][word]=np.nanmean(AT_latencyAmplitudeAnalysis['amplitude_word'][word])
        word_lat_amp_results_detail[participant][1][3][word]=np.nanstd(AT_latencyAmplitudeAnalysis['amplitude_word'][word])
    for speaker in range(4):     
        speaker_lat_amp_results_detail[participant][1][0][speaker]=np.nanmean(AT_latencyAmplitudeAnalysis['latency_word'][speaker])
        speaker_lat_amp_results_detail[participant][1][1][speaker]=np.nanstd(AT_latencyAmplitudeAnalysis['latency_word'][speaker])
        speaker_lat_amp_results_detail[participant][1][2][speaker]=np.nanmean(AT_latencyAmplitudeAnalysis['amplitude_word'][speaker])
        speaker_lat_amp_results_detail[participant][1][3][speaker]=np.nanstd(AT_latencyAmplitudeAnalysis['amplitude_word'][speaker])

#%%
#paird_ttest_word=np.full((12,8,4),np.nan)# 12 participants, 8 words, 4 quantities(t and p value for latency and amplitude )
#paird_ttest_speaker=np.full((12,4,4),np.nan)
#for p in range(12):
#    for w in range(8):
#            paird_ttest_word[p][w][0]=scipy.stats.ttest_rel(word_lat_amp_results_detail[p][0][0][w],word_lat_amp_results_detail[p][1][0][0][w])[0]
#            paird_ttest_word[p][w][1]=scipy.stats.ttest_rel(word_lat_amp_results_detail[p][0][0][w],word_lat_amp_results_detail[p][1][0][0][w])[1]
#            paird_ttest_word[p][w][2]=scipy.stats.ttest_rel(word_lat_amp_results_detail[p][0][2][w],word_lat_amp_results_detail[p][1][0][2][w])[0]
#            paird_ttest_word[p][w][3]=scipy.stats.ttest_rel(word_lat_amp_results_detail[p][0][2][w],word_lat_amp_results_detail[p][1][0][2][w])[1]
#            
#    
#    for sp in range(4):
#            paird_ttest_speaker[p][w][0]=scipy.stats.ttest_rel(speaker_lat_amp_results_detail[p][0][0][w],speaker_lat_amp_results_detail[p][1][0][0][w])[0]
#            paird_ttest_speaker[p][w][1]=scipy.stats.ttest_rel(speaker_lat_amp_results_detail[p][0][0][w],speaker_lat_amp_results_detail[p][1][0][0][w])[1]
#            paird_ttest_speaker[p][w][2]=scipy.stats.ttest_rel(speaker_lat_amp_results_detail[p][0][2][w],speaker_lat_amp_results_detail[p][1][0][2][w])[0]
#            paird_ttest_speaker[p][w][3]=scipy.stats.ttest_rel(speaker_lat_amp_results_detail[p][0][2][w],speaker_lat_amp_results_detail[p][1][0][2][w])[1]
#            
#        

#%%
word_latency_meanOfAllParticipants=np.nanmean(np.reshape(word_lat_amp_results_detail,(22,4,8))[:,0,:],axis=0)
word_latency_stdOfAllParticipants=np.nanstd(np.reshape(word_lat_amp_results_detail,(22,4,8))[:,1,:],axis=0)

word_amplitude_meanOfAllParticipants=np.nanmean(np.reshape(word_lat_amp_results_detail,(22,4,8))[:,2,:],axis=0)
word_amplitude_stdOfAllParticipants=np.nanstd(np.reshape(word_lat_amp_results_detail,(22,4,8))[:,3,:],axis=0)

speaker_latency_meanOfAllParticipants=np.nanmean(np.reshape(speaker_lat_amp_results_detail,(22,4,4))[:,0,:],axis=0)
speaker_latency_stdOfAllParticipants=np.nanstd(np.reshape(speaker_lat_amp_results_detail,(22,4,4))[:,1,:],axis=0)
    
speaker_amplitude_meanOfAllParticipants=np.nanmean(np.reshape(speaker_lat_amp_results_detail,(22,4,4))[:,2,:],axis=0)
speaker_amplitude_stdOfAllParticipants=np.nanstd(np.reshape(speaker_lat_amp_results_detail,(22,4,4))[:,3,:],axis=0)
