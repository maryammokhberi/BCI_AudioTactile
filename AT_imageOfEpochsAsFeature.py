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
#import AT_FeatureExtraction



#from autoreject import LocalAutoRejectCV
bads=['PO7','PO8','T7','T8','C3','C4','F3','F4','STI 014'] 
#%% change working directory and set parameters

path="E:\Biomedical.master\Data\par130_1" #TODO : go back to input design
os.chdir(path)
montage=mne.channels.read_montage('standard_1020')
layout=mne.channels.read_layout('EEG1005')
layout2=mne.channels.read_layout('EGI256')
blockNum=5
runNum=10
samp_freq=float(1000)
resamp_freq=float(1000)

#%% load data files

fname="E:\Biomedical.master\Data\par130_1\AudioTactile.vhdr" #TODO: go back to input design
# load the data from matlab interdace: sequence of stimuli, oddball stimuli, 
# and correct responses
results_all= scipy.io.loadmat("results.mat")
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
# saving annotation step
#onset=[325,366.1,385.384,416.6,429,452,462.4,480.2,508,518,534.16,598.1,1206,\
#        1215.1,1233.2,1257.5,1298,1327.8,1420.2,1424,1430.66,1443.7,1935.6,\
#        1949.12,1989,2010,2502.5,2508,2514,2526.4,2568.1,2719.3,2721.6,2738.8,\
#        2744.4,2883.7,2916,3278.1]
#duration=[1.8,1.4,1,1,1,0.8,0.9,3,1,1,.7,0.9,.5,1.6,.4,0.45,1.9,0.8,1,.45,\
#            0.45,1.2,.75,.6,.450,1.5,1.5,2,1.5,.45,1,.5,1.5,3,1.2,1,3,1.7]
#badchannels=['CP4','Oz','F3']
#annot_params=dict(onset=onset,duration=duration,badchannels=badchannels)
#np.save('annot_params.npy',annot_params)

#loading annotation step
annot_params= np.load('annot_params.npy')
onset=annot_params.item().get('onset')
duration=annot_params.item().get('duration')
badchannels=annot_params.item().get('badchannels')
annotations=mne.Annotations(onset,duration,'bad')
AudioTactile.annotations=annotations
bads.extend(badchannels)
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
#end of session(=end of last block) will be definded 10 seconds after last stimulus event is last block
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
del AudioTactile 
del blockReplica
del temprow

#%% preliminary plot of epochs and feature extraction

#pipeline: downsampling, filtering 1,12 ,re-reference [], epoch, baseline -.4,
#          reject bads, moving average decim?40, peak picking, alignment, 
#          average, normalization, windsoring, detrend


numOfChans=17
numOfBestChans=numOfChans- len(bads) #one channel is stim (marker) channel
numOfFeatures=41
numOfTrainBlocks=5
tmin=0
tmax=.8
decim=20
X=np.full([numOfTrainBlocks,runNum,8,12,numOfBestChans,numOfFeatures],np.nan)
y=np.full([numOfTrainBlocks,runNum,8,1],np.nan)
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
        trial.filter(0.5,12,method='iir')   
        trial.resample(sfreq=resamp_freq)
        trial_Epoch=mne.Epochs(trial_rerefrenced, exported_events, tmin=-.4,baseline=(-.16,None),
                               tmax=1.5, decim=decim, reject_by_annotation=True) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate
        
        trial_Epoch.load_data()
        trial_Epoch.crop(tmin=tmin,tmax=tmax)      
        
        trial_Epoch.drop_channels(trial_Epoch.info['bads'])
        trial_Epoch_data=trial_Epoch.get_data()   
        #trial_Epoch.drop_bad() 

        
        
        
        
        trial_x=np.zeros( (104, numOfBestChans*numOfFeatures), dtype=float)
        trial_y=np.zeros((104,1))
        
        for s in range(8) :
            stimulus_code_str=str(s+8)
            stim_epochs=trial_Epoch[stimulus_code_str]
            stim_epochs_data=stim_epochs.get_data()
            oddball=oddball_stim[b,0][r,0] #the stimcode starts from 1 here. to get the event stim, add 7. e.g. 1--> 8: cow
            #since some epochs will be deleted due to bad data, we need to substitute some data instead of them. we choose the mean of other epochs
            if (stim_epochs_data.shape[0]<12) :
                stim_epochs_data_mean=stim_epochs_data.mean(axis=0)
                stim_epochs_data=np.append(stim_epochs_data,[stim_epochs_data_mean]*(12-stim_epochs_data.shape[0]), axis=0)
             
            for e in range(12):  #just ignore the 13th epoch to keep matrix same-length for all stim codes
                
#                if e>(stim_epochs_data.shape[0]-1):
#                    break
                
                oddball=oddball_stim[b,0][r,0]
                if s+1==oddball:
                   y[b][r][s][0]=1
                else:
                   y[b][r][s][0]=0
              
                 
                for c in range(numOfBestChans):
                                     
                    #assigning the features to a feature vector
                    X[b][r][s][e][c][0:numOfFeatures]=stim_epochs_data[e][c]
                    #keeping a version of multi-dimensional X before reshaping it
                    X_multi_D=X
            
        
            
        
        


X_avgOfChans=X.mean(axis=4)
#reshaping X into a 2D array: n_data(50*8=400) and n_features(12*1000/decim)
X=np.reshape(X_avgOfChans,(numOfTrainBlocks*runNum*8,12*numOfFeatures)) 
y=np.reshape(y,(numOfTrainBlocks*runNum*8,1))
#X_avgChans=np.mean(X_multi_D,axis=4)           
#X_avgChans=np.reshape(X_avgChans,(numOfTrainBlocks*runNum*8*13,numOfFeatures))


gc.collect()

#%%

b=0
r=0
print "********preparing data for the stimuli in block", b+1,"run#", r+1,"****************\
        *********************"
trial=AudioTactile_runs[b][r]
trial.n_times
trial.load_data()
trial.info['bads']=bads
trial.set_montage(montage)
trial_rerefrenced, _= mne.set_eeg_reference(trial,[])
trial.filter(1,12,method='iir')   
trial.resample(sfreq=resamp_freq)
trial_Epoch=mne.Epochs(trial_rerefrenced, exported_events, tmin=-.4,baseline=(-.16,None),
                       tmax=1.5, decim=decim, reject_by_annotation=True) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate

trial_Epoch.load_data()
trial_Epoch.crop(tmin=tmin,tmax=tmax)      

trial_Epoch.drop_channels(trial_Epoch.info['bads'])
oddball_check=oddball_stim[b,0][r,0]


trial_Epoch['8'].plot_image(combine='mean')
trial_Epoch['9'].plot_image(combine='mean')
trial_Epoch['10'].plot_image(combine='mean')
trial_Epoch['11'].plot_image(combine='mean')
trial_Epoch['12'].plot_image(combine='mean')
trial_Epoch['13'].plot_image(combine='mean')
trial_Epoch['14'].plot_image(combine='mean')
trial_Epoch['15'].plot_image(combine='mean')




