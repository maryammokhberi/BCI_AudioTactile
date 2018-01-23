# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:21:51 2017

@author: Amarantine
"""

#%% import relevant toolboxes
#clear all?
get_ipython().magic('reset -sf')
from IPython import get_ipython

import mne # a toolbox to use work with bio-signals
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import math
import gc
plt.close('all')
import AT_FeatureExtraction



#from autoreject import LocalAutoRejectCV



#%% change working directory and set parameters

path="E:\Biomedical.master\Data\par011_2" #TODO : go back to input design
os.chdir(path)
montage=mne.channels.read_montage('standard_1020')
layout=mne.channels.read_layout('EEG1005')
layout2=mne.channels.read_layout('EGI256')
blockNum=5
runNum=10
samp_freq=float(1000)
resamp_freq=float(1000)

#%% load data files

fname=os.path.join(path, "AudioTactile.vhdr") #TODO: go back to input design
# load the data from matlab interdace: sequence of stimuli, oddball stimuli, 
# and correct responses
results_all= scipy.io.loadmat(os.path.join(path, 'results.mat'))
results_firstInChain=results_all['results']
results=results_firstInChain[0] 
sequence_stim=results[0][0]
#an object of 5,1, containing 5 object of 10,1; each containing a sequence \
# of 100, of the auditory stimulus
oddball_stim=results[0][1]
#an object of 5,1, containing 5 object of 10,1; containg the oddball stimulus\
# for each run
response_is_correct=results[0][2]
#an object of 5,1,10 containing 5 object of 1,110; containing binary data of \
#the correctness of the particpants response at the end of each run

gc.collect()

#%%
AudioTactile= mne.io.read_raw_brainvision (fname,preload=True)

gc.collect() 


#%% add event list to the brainvision raw object
exported_events=np.load("exported_events.npy")  # have manipulated \
#brainvision.py to save events as exported_events.npy from  read marker files
exported_events[:,0]=exported_events[:,0]/(samp_freq/resamp_freq)     
events=exported_events.tolist()
AudioTactile.add_events(events)
gc.collect()

#%%
#AudioTactile.notch_filter(np.arange(60, 302, 120), filter_length='auto')
AudioTactile.resample(sfreq=resamp_freq) 
#Tim zeyl used the range 0.3, 20 Hz for filetering range. Erwei used 0.1,45 HZ 
#as filtring range. P300 info is dominant in 0.1-4 HZ. Tim used 4th order butterworth (IIR)
#and Erwei did not mention the type of filter. Alborzused FIR for his p300 program.
AudioTactile.filter(0.5,12,None,method='iir') 
AudioTactile.info['lowpass']=12



 #%%annotations 
###### saving annotation step
#onset=[191,529,543,552,610.8,680,838,915,1014.5,1145.5,1248,1275,1289.5,1302,
#        1320,1328,1348,1358,1508,1519,1595,1610,1616,1636.5,1654,1718.5,1716,
#        1727.3,1831.5,1876,1890,1900,1915,1918,1926,1964,2012.5,2025.5,2037,
#        2784,2820,2882,2894,2920,2950.5,2795,3014,3123,3137.5,3170,3178,3194,
#        3202,3245,3262,3478,3508,3521,3572,3580,3592.5,3625,3632,3638,3651,3682,
#        3728,3787,3800,3820,3880,3940] 
#duration=[3,2,3,1,1,1.5,2,3,1.5,1.5,4,1,1.5,5,2,1.5,2,10,2,2,3,2,3,1.5,1,1.5,1,
#           1,1.5,2,1.5,1.5,2,6,2,5,1.5,2,3,2,2,2,2,6,1.5,5,7,1.5,3,2,4,5,2,5,3,
#           6,2,3,2.5,1,1.5,2.5,2,3,3,2,2,2,10,2,3,28]
##############
#badchannels=[ 'Oz','PO8','PO7','CP4','T8','T7','F3','F4', 'STI 014']
#annot_params=dict(onset=onset,duration=duration,badchannels=badchannels)
#np.save(os.path.join(path, 'annot_params.npy'), annot_params)

#loading annotation step
annot_params= np.load(os.path.join(path, 'annot_params.npy'))
onset=annot_params.item().get('onset')
duration=annot_params.item().get('duration')
badchannels=annot_params.item().get('badchannels')
annotations=mne.Annotations(onset,duration,'bad')
AudioTactile.annotations=annotations
bads=badchannels
AudioTactile.info['bads']=bads



#gc.collect()
#%% visualize event
event_id={'Baseline Start':1,'Baseline stop':2,'run start':6,'Starto''cow': 8,\
          'frog': 9,
            'bear': 10, 'mouse': 11,
            'cat': 12, 'chick': 13, 'fox':14 , 
            'horse':15}
mne.viz.plot_events(events, AudioTactile.info['sfreq'], event_id=event_id)
#%% separate blocks 
event_time_baseline=[exported_events[i,0] for i in range(len(exported_events))\
                     if exported_events[i,2] == 1]
time_endOfExp=exported_events[len(exported_events)-1, 0]+10*resamp_freq \
#end of session(=end of last block) will be definded 10 seconds after last stimulus event is lasbat block
AudioTactile_blocks=[None]*blockNum

# create blocks : a dictionary which its  values are cropped brainvision raw 
# objects
for i in range(blockNum):
#    print "preparing block#", i+1
    AudioTactileReplica = AudioTactile.copy()
#    print "Whole audio-tactile task length in sec", \
#        AudioTactileReplica.n_times / resamp_freq
    tmin = event_time_baseline[i] / resamp_freq                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    if i < blockNum - 1:                         
        tmax=event_time_baseline[i+1]/resamp_freq
    else:
        tmax=time_endOfExp/resamp_freq  # end of session(=end of last block) \
        #will be definded 10 seconds after last stimulus event is last block                   
    AudioTactile_blocks[i] = AudioTactileReplica.crop(tmin=tmin, tmax=tmax)
#    print "block length in sec:", AudioTactile_blocks[i].n_times / resamp_freq
                        
    del AudioTactileReplica   
    

gc.collect()    

#%% separate runs 
event_time_runs=[exported_events[i,0] for i in range(len(exported_events)) if \
                 exported_events[i,2] == 6]
time_endOfExp=exported_events[len(exported_events)-1, 0]+10*resamp_freq
AudioTactile_runs=[[None]*runNum]*blockNum
blockReplica=[None]*blockNum
             
for i in range(runNum*blockNum):
    #preparing variables for the dictionary 
    block=int(math.floor(i/runNum)) #variable block : actual block number -1
    run=i%10                   #variable run : actual run number -1     
#    print "********preparing block", block+1,"run#", run+1,"****************\
#    *********************"  
    #making a replica of the AudioTactile_blocks, so we can crop it without \
 
    blockReplica[block]=AudioTactile_blocks[block].copy()

    #setting the tmin and tmax: using the timing of events
    tmin=(event_time_runs[i] - event_time_baseline[block])/resamp_freq

    if (i%10)<9:                         
         tmax=(event_time_runs[i+1] - event_time_baseline[block])/resamp_freq

    else:
        tmax=None
        
    temprow=AudioTactile_runs[block][:]    
    temprow[run] = blockReplica[block].crop(tmin=tmin, tmax=tmax) # cropping changes the size of primary objetc; here blockReplica[block]
    AudioTactile_runs[block]=temprow #remember how assigning a value to list of list affects other values in the list. i.e.: list[i][j]=VALUE changes all the values in column j to VALUE

#    print "AudioTactile_runs",block+1,",",run+1," length in sec" ,\
#                            AudioTactile_runs[block][run].n_times/resamp_freq

gc.collect()    


#%% delete unnecessary variables for the sake of freeing memory
del AudioTactile_blocks
#del AudioTactile 
del blockReplica
del temprow

#%% preliminary plot of epochs and feature extraction

#pipeline: downsampling, filtering 1,12 ,re-reference [], epoch, baseline -.4,
#          reject bads, moving average decim?40, peak picking, alignment, 
#          average, normalization, windsoring, detrend


numOfChans=17
numOfBestChans=numOfChans- len(bads) #one channel is stim (marker) channel
numOfFeatures=4
numOfTrainBlocks=blockNum
tmin=0.1
tmax=.8
decim=20
oddball_X=np.full([numOfTrainBlocks,runNum,12,numOfBestChans,numOfFeatures],np.nan)
editedRuns=[]
numOfRejectedEpochs=[]
#%%

for b in range(numOfTrainBlocks):
    
    for r in range(runNum):


        print "********preparing data for the stimuli in block", b+1,"run#", r+1,"****************\
        *********************"
        trial=AudioTactile_runs[b][r]
        trial.n_times
        trial.load_data()
        trial.info['bads']=bads
        trial.set_montage(montage)
        trial_rerefrenced, _= mne.set_eeg_reference(trial,[])
#        trial.filter(0.5,12,method='iir')   
        trial.resample(sfreq=resamp_freq)
        trial_Epoch=mne.Epochs(trial_rerefrenced, exported_events, tmin=-.4,baseline=(-.16,None),
                               tmax=1.5, decim=decim, reject_by_annotation=True) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate
        
        trial_Epoch.load_data()
        trial_Epoch.crop(tmin=tmin,tmax=tmax)      
        
        trial_Epoch.drop_channels(trial_Epoch.info['bads'])

        
        
        #trial_Epoch.drop_bad() 

        oddball_str=str(oddball_stim[b,0][r,0]+7)
        oddball_epochs=trial_Epoch[oddball_str]
        oddball_epochs_data=oddball_epochs.get_data()
        oddball=oddball_stim[b,0][r,0] 
        
        if (oddball_epochs_data.shape[0]<12) :
            break
            
        for e in range(12):  #just ignore the 13th epoch to keep matrix same-length for all stim codes        
            for c in range(numOfBestChans):
                
                                    #extract featues for each epoch
                    #0
                    latency=AT_FeatureExtraction.latency(oddball_epochs_data[e][c], tmin, decim)
                    #1
                    amplitude=AT_FeatureExtraction.amplitude(oddball_epochs_data[e][c])
                    #2
                    speakerNumber=np.remainder(np.argwhere(sequence_stim[b,0][r,0]==oddball)[e,0],4)+1
                    #3
                    oddball_name=oddball
                    #assigning the features to a feature vector
                    oddball_X[b][r][e][c][0:4]=np.concatenate(([latency],[amplitude],[speakerNumber],[oddball_name]))
               

                    #keeping a version of multi-dimensional X before reshaping it
                    oddball_X_multi_D=oddball_X
            

#averaging over the channels
oddball_X_avgOfChans=oddball_X.mean(axis=3)
oddball_X=np.reshape(oddball_X_avgOfChans,(oddball_X_avgOfChans.shape[0]*runNum,12,numOfFeatures)) 

#deleting nan elements
nan_elements=np.argwhere(np.isnan(oddball_X)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
oddball_X =np.delete(oddball_X, nan_index, axis=0)

oddball_X=np.reshape(oddball_X,(oddball_X.shape[0]*oddball_X.shape[1],numOfFeatures))
np.save(os.path.join(path, 'oddball_X.npy'),oddball_X)



gc.collect()
#%%
latency_word=[None]*8 #aoddball latency based on stimulus word
latency_word_mean=[None]*8
latency_word_std=[None]*8
amplitude_word=[None]*8 #oddball amplitude based in stimulus word
amplitude_word_mean=[None]*8
amplitude_word_std=[None]*8
for o in range(8):
    latency_word[o]=oddball_X[np.argwhere(oddball_X[:,3]-1==o),0]
    latency_word_mean[o]=np.mean(latency_word[o])
    latency_word_std[o]=np.std(latency_word[o]) 
    amplitude_word[o]=oddball_X[np.argwhere(oddball_X[:,3]-1==o),1]
    amplitude_word_mean[o]=np.mean(amplitude_word[o])
    amplitude_word_std[o]=np.std(amplitude_word[o])
    
latency_speaker=[None]*4 #oddball latency based on speaker position
latency_speaker_mean=[None]*4
latency_speaker_std=[None]*4
amplitude_speaker=[None]*4 #oddball amp based on speaker positions
amplitude_speaker_mean=[None]*4
amplitude_speaker_std=[None]*4
for speakerIndex in range(4):
    latency_speaker[speakerIndex]=oddball_X[np.argwhere(oddball_X[:,2]-1==speakerIndex),0]
    latency_speaker_mean[speakerIndex]=np.mean(latency_speaker[speakerIndex])
    latency_speaker_std[speakerIndex]=np.std(latency_speaker[speakerIndex])
    amplitude_speaker[speakerIndex]=oddball_X[np.argwhere(oddball_X[:,2]-1==speakerIndex),1]
    amplitude_speaker_mean[speakerIndex]=np.mean(amplitude_speaker[speakerIndex])
    amplitude_speaker_std[speakerIndex]=np.std(amplitude_speaker[speakerIndex])

print "latency_word_mean",latency_word_mean
print "amplitude_word_mean",amplitude_word_mean        
print "latency_speaker_mean",latency_speaker_mean
print "amplitude_speaker_mean",amplitude_speaker_mean 

AT_latencyAmplitudeAnalysis=dict(latency_word=latency_word,
                                 latency_word_mean=latency_word_mean,
                                 latency_word_std=latency_word_std,
                                 amplitude_word=amplitude_word,
                                 amplitude_word_mean=amplitude_word_mean,
                                 amplitude_word_std=amplitude_word_std,
                                 latency_speaker=latency_speaker,
                                 latency_speaker_mean=latency_speaker_mean,
                                 latency_speaker_std=latency_speaker_std,
                                 amplitude_speaker=amplitude_speaker,
                                 amplitude_speaker_mean=amplitude_speaker_mean,
                                 amplitude_speaker_std=amplitude_speaker_std)

import pickle
with open('AT_latencyAmplitudeAnalysis', 'wb') as fp:
    pickle.dump(AT_latencyAmplitudeAnalysis, fp)



#%%
