# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:14:22 2017 

@author: Maryam Mokheri
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
import AT_FeatureExtraction

#from autoreject import LocalAutoRejectCV
bads=['CP4','T7','T8'] 
#%% change working directory and set parameters

#path= input("Please enter the path for the data from audioTactile oddball \
#task: ")
path="E:\Biomedical.master\Data\par090_1" #TODO : go back to input design
os.chdir(path)
montage=mne.channels.read_montage('standard_1020')
layout=mne.channels.read_layout('EEG1005')
layout2=mne.channels.read_layout('EGI256')
blockNum=5
runNum=10
samp_freq=float(1000)
resamp_freq=float(1000)

#%% load data files

#fname= input("Please enter the path for the data from audioTactile oddball \
#task:\n Please include the filename in the path and make sure to put double \
#quotes around the it.") #please enter the path for the data
fname="E:\Biomedical.master\Data\par090_1\AudioTactile.vhdr" #TODO: go back to input design
#file_name= input("Please enter the name of the data from audioTactile oddball \
# task: ")
#fname= [file_path, '/' ,file_name]

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
AudioTactile.info['bads']=bads
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
AudioTactile.filter(1,12,method='iir') 
AudioTactile.info['lowpass']=12
#%%annotations 
annot_params= np.load('annot_params.npy')
onset=annot_params[0]['onset']
duration=annot_params[0]['duration']
annotations=mne.Annotations(onset,duration,'bad')
AudioTactile.annotations=annotations


gc.collect()
#%% preliminary data check

AudioTactile.n_times
#AudioTactile_data= AudioTactile.get_data()
#last_chan=AudioTactile_data[16,:]
#print np.sum(last_chan)

#%% Epoch the data 
#AudioTactile_Epoch=mne.Epochs(AudioTactile, exported_events, tmin=-.1, \
#tmax=.55)
#AudioTactile_Epoch_data=AudioTactile_Epoch.get_data()
#AudioTactile_Epoch.plot_image(9, cmap='interactive')
#AudioTactile_Epoch_runs=mne.Epochs(AudioTactile, exported_events, event_id={'run start':6}, tmin=-1, tmax=60)

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
    print "preparing block#", i+1
    AudioTactileReplica = AudioTactile.copy()
    print "Whole audio-tactile task length in sec", \
        AudioTactileReplica.n_times / resamp_freq
    tmin = event_time_baseline[i] / resamp_freq                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    if i < blockNum - 1:                         
        tmax=event_time_baseline[i+1]/resamp_freq
    else:
        tmax=time_endOfExp/resamp_freq  # end of session(=end of last block) \
        #will be definded 10 seconds after last stimulus event is last block
    #block=i+1                       
    AudioTactile_blocks[i] = AudioTactileReplica.crop(tmin=tmin, tmax=tmax)
    print "block length in sec:", AudioTactile_blocks[i].n_times / resamp_freq
#    print tmin
#    print tmax 
#    print AudioTactile_blocks[i].n_times
#    print AudioTactile_blocks[0].n_times                         
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
    print "********preparing block", block+1,"run#", run+1,"****************\
    *********************"
#    print "run:", run
#    print "block:", block
    
    #making a replica of the AudioTactile_blocks, so we can crop it without \
#    affecting the AudioTactile_blocks   
    blockReplica[block]=AudioTactile_blocks[block].copy()
#    print blockReplica[block]
#    print AudioTactile_blocks[block].copy()
#    print "block copy length in sec:", blockReplica[block].n_times /resamp_freq
    #setting the tmin and tmax: using the timing of events
    tmin=(event_time_runs[i] - event_time_baseline[block])/resamp_freq
#    print "tmin:", tmin, "event_time_runs[i]/resamp_freq:", event_time_runs[i]/resamp_freq, "event_time_baseline[block])/resamp_freq:", event_time_baseline[block]/resamp_freq
    if (i%10)<9:                         
         tmax=(event_time_runs[i+1] - event_time_baseline[block])/resamp_freq
#         print "tmax:", tmax, "event_time_runs[i+1]:",event_time_runs[i+1]/resamp_freq,"event_time_baseline[block])/resamp_freq:", event_time_baseline[block]/resamp_freq
#    elif (i%10==9 and block==blockNum-1) : #in the last run of last block, assign tmax=time_endOfExp        
#         tmax=(tmin+ 60*resamp_freq)/resamp_freq # assuming the last run is about 60 sec
    else:
        tmax=None
        
    temprow=AudioTactile_runs[block][:]    
    temprow[run] = blockReplica[block].crop(tmin=tmin, tmax=tmax) # cropping changes the size of primary objetc; here blockReplica[block]
    AudioTactile_runs[block]=temprow #remember how assigning a value to list of list affects other values in the list. i.e.: list[i][j]=VALUE changes all the values in column j to VALUE
#    print "run length in sec: ", AudioTactile_runs[block][run].n_times/resamp_freq
#    print event_time_baseline[block]/resamp_freq
#    print event_time_runs[i]/resamp_freq
#    if (i%10)<9: print event_time_runs[i+1]/resamp_freq 
#    print tmin
#    print tmax
#    if (i%10)<9: print "(tmax-tmin)* resamp_freq:" , (tmax-tmin)* resamp_freq
##    print "AudioTactile_runs[",block,"][",run,"].n_times:" , AudioTactile_runs[block][run].n_times
    print "AudioTactile_runs",block+1,",",run+1," length in sec" ,\
                            AudioTactile_runs[block][run].n_times/resamp_freq
#    if i>0: print "AudioTactile_runs[0][1] length in sec" , \
#                                    AudioTactile_runs[0][1].n_times/resamp_freq
#    if i>9: print "AudioTactile_runs[1][0] length in sec" , \
#                                    AudioTactile_runs[1][0].n_times/resamp_freq
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
 
#TODO: Normalize data and reject bad channels and epochs
#TODO: windsoring the trials
#TODO: moving average
#TODO: re-reference the electodes
#TODO: check low pass filtering
#TODO: reduce baseline
#TODO: increase post-stimulus time
#TODO: peak-picking and alignment

#bestChans=np.array([False, False, False, True, False, False, False, True, True, \
#                            False, False, False, False, False, False, False])
bads=['PO8','PO7','P3','P4','C3','C4','CP3','CP4','F3','F4','T7','T8','STI 014'] #non-relevant channels
numOfChans=17
numOfBestChans=numOfChans- len(bads) #one channel is stim (marker) channel
numOfFeatures=17
numOfTrainBlocks=5
tmin=.4
tmax=.8
decim=20
X=np.full([numOfTrainBlocks,runNum,8,13,numOfBestChans,numOfFeatures],np.nan)
y=np.full([numOfTrainBlocks,runNum,8,13,1],np.nan)
bad_epochs=np.zeros((numOfTrainBlocks, runNum))
prediction_is_correct=np.zeros((50,1))
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
        trial.filter(1,12,method='iir')
        trial.resample(sfreq=resamp_freq)
        trial_Epoch=mne.Epochs(trial_rerefrenced, exported_events, tmin=-.4,
                               tmax=1.5, decim=decim, reject_by_annotation=True,
                               reject=dict(eeg=2e-4)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate

        trial_Epoch.load_data()
        trial_Epoch_long=trial_Epoch.copy() # to be used later in peak-alignment
        trial_Epoch.crop(tmin=tmin,tmax=tmax)        
        trial_Epoch.drop_channels(trial_Epoch.info['bads'])
        trial_Epoch_data=trial_Epoch.get_data()
        #######creat feature vector for all stimuli in one run(trial) ########        
        ##trial_Epoch.drop_bad() 

        trial_x=np.zeros( (104, numOfBestChans*numOfFeatures), dtype=float)
        trial_y=np.zeros((104,1))
        
        for s in range(8) :
            stimulus_code_str=str(s+8)
            stim_epochs=trial_Epoch[stimulus_code_str]
            stim_epochs_data=stim_epochs.get_data()
            oddball=oddball_stim[b,0][r,0] #the stimcode starts from 1 here. to get the event stim, add 7. e.g. 1--> 8: cow
            for e in range(13):
                
                if e>(stim_epochs_data.shape[0]-1):
                    break
                
                oddball=oddball_stim[b,0][r,0]
                if s+1==oddball:
                   y[b][r][s][e][0]=1
                else:
                   y[b][r][s][e][0]=0
              
                 
                for c in range(numOfBestChans):
                    #extract featues for each epoch
                    #0
                    latency=AT_FeatureExtraction.latency(stim_epochs_data[e][c], tmin, decim)
                    #1
                    amplitude=AT_FeatureExtraction.amplitude(stim_epochs_data[e][c])
                    #2
                    lat_amp_ratio=AT_FeatureExtraction.lat_amp_ratio(stim_epochs_data[e][c], tmin, decim)
                    #3
                    abs_amp=AT_FeatureExtraction.abs_amp(stim_epochs_data[e][c])
                    #4
                    abs_lat_amp_ratio=AT_FeatureExtraction.abs_lat_amp_ratio(stim_epochs_data[e][c], tmin, decim)
                    #5
                    positive_area=AT_FeatureExtraction.positive_area(stim_epochs_data[e][c])
                    #6
                    negative_area=AT_FeatureExtraction.negative_area(stim_epochs_data[e][c])
                    #7
                    total_area=AT_FeatureExtraction.total_area(stim_epochs_data[e][c])
                    #8
                    abs_total_area=AT_FeatureExtraction.abs_total_area(stim_epochs_data[e][c])
                    #9
                    total_abs_area=AT_FeatureExtraction.total_abs_area(stim_epochs_data[e][c])
                    #10
                    avg_abs_slope=AT_FeatureExtraction.avg_abs_slope(stim_epochs_data[e][c])
                    #11
                    peak_to_peak=AT_FeatureExtraction.peak_to_peak(stim_epochs_data[e][c])
                    #12
                    pk_to_pk_tw=AT_FeatureExtraction.pk_to_pk_tw(stim_epochs_data[e][c],decim)
                    #13
                    pk_to_pk_slope=AT_FeatureExtraction.pk_to_pk_slope(stim_epochs_data[e][c],decim)
                    #14
                    zero_cross=AT_FeatureExtraction.zero_cross(stim_epochs_data[e][c])
                    #15
                    zero_cross_density=AT_FeatureExtraction.zero_cross_density(stim_epochs_data[e][c],decim)
                    #16
                    slope_sign_alt=AT_FeatureExtraction.slope_sign_alt(stim_epochs_data[e][c])
                            
                    #assigning the features to a feature vector
                    X[b][r][s][e][c][0:17]=[latency,amplitude,lat_amp_ratio,
                                             abs_amp,abs_lat_amp_ratio,positive_area,
                                             negative_area,total_area,abs_total_area,
                                             total_abs_area,avg_abs_slope,peak_to_peak,
                                             pk_to_pk_tw,pk_to_pk_slope,zero_cross,
                                             zero_cross_density,slope_sign_alt]
               
                
                
    


#reshaping X into a 2D array: n_data and n_features
X=np.reshape(X,(numOfTrainBlocks*runNum*8*13,numOfBestChans*numOfFeatures)) 
y=np.reshape(y,(numOfTrainBlocks*runNum*8*13,1))           


        
#oversampling the oddball stimuli to fix imabalance of the data
Xy=np.concatenate((X,y),axis=1)
Xy_balanced=np.zeros(((Xy.shape[0])*14/8,Xy.shape[1]), dtype=float) # 14 is 7 non-oddball + 7*1 oddball data 
j=0
for i in range(Xy.shape[0]):
    if Xy[i,-1]==1:
        Xy_balanced[j:j+7,:]=np.tile(Xy[i,:], (7,1))
        j=j+7
        
    else:
        Xy_balanced[j,:]=Xy[i,:]
        j=j+1




X_balanced=Xy_balanced[:,0:-1]        
y_balanced=Xy_balanced[:,-1]

Xy_oddball=Xy[np.argwhere(Xy[:,-1]==1)]
Xy_nonoddball=Xy[np.argwhere(Xy[:,-1]==0)]
                    
                    
                    
#trial_Epoch[oddball_str].plot_image(8, cmap='interactive')
##trial_Epoch.apply_baseline()
##trial_Epoch.plot_psd
##trial_Epoch.plot_sensors
##trial_Epoch.plot_topo_image
##trial_Epoch.plot_psd_topomap
gc.collect()


#%% plot the test run and epoch data
#
#b=1
#r=0
#        
#print "********preparing data for the stimuli in block", b+1,"run#", r+1,"****************\
#        *********************"
#trial=AudioTactile_runs[b][r]
#trial.n_times
#trial.load_data()
#trial.info['bads']=bads
#trial.set_montage(montage)
#trial_rerefrenced, _= mne.set_eeg_reference(trial,[])
#trial.filter(1,12,method='iir')
#trial.resample(sfreq=resamp_freq)
#trial_Epoch=mne.Epochs(trial_rerefrenced, exported_events, tmin=-.4,
#                   tmax=1.5, decim=5, reject_by_annotation=True,
#                   reject=dict(eeg=2e-4)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate
#trial_Epoch.load_data()
#trial_Epoch_long=trial_Epoch.copy() # to be used later in peak-alignment
#trial_Epoch.crop(tmin=.18,tmax=.7)        
##        ar = LocalAutoRejectCV()
##        epochs_clean = ar.fit_transform(test_Epoch)
#
##        trial_Epoch.plot()
##        plt.ion()
##        plt.pause(30)
##        bad_epochs[b][r]=raw_input("Please mark bad epochs on the plot and enter them as a list of ints here")
#trial_Epoch.drop_channels(trial_Epoch.info['bads'])
##        trial_Epoch.plot_image(combine='mean', cmap='interactive')
##trial_Epoch.drop_bad() # drops bad epochs automatically
#trial_Epoch_data=trial_Epoch.get_data()
########creat feature vector for all stimuli in one run(trial) ########        
###trial_Epoch.drop_bad() 
#trial_x=np.zeros( (8, numOfBestChans*trial_Epoch.times.shape[0] ), dtype=float)
#trial_y=np.zeros((8,1))
#
#for i in range(8) :
#    stimulus_code_str=str(i+8)
#    #plot average of stim_code epochs voltage + cmap of individuals    
#    #trial_Epoch[stimulus_code_str].plot_image(4, cmap='interactive')
#
#    
#    #average over stim epochs
#    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
#    #stimulus_code_avg.plot()
#    stimulus_code_avg_data = stimulus_code_avg.data
#    stimulus_code_avg_data_norm =\
#        stimulus_code_avg_data/np.max( 
#                np.abs(stimulus_code_avg_data)) #normalization
#     
#        
#    #original signal as feature (channels concatenared after each other)
#    stimulus_code_concatenatedBestChans=stimulus_code_avg_data_norm.reshape(
#        np.size(stimulus_code_avg_data_norm))
#    
#    #r-squared as feature 
#    
#    
#    stimulus_code_rsquared= np.square((stimulus_code_avg_data- np.mean(\
#                                          stimulus_code_avg_data\
#                                          )))/(float(np.size(stimulus_code_avg_data))*\
#                                            np.var(stimulus_code_avg_data))
#        #(channels concatenared after each other)
#    stimulus_code_rsquared_concatenated=stimulus_code_rsquared.reshape(
#            np.size(stimulus_code_rsquared))
#        #Average of channels 
#    stimulus_code_rsquared_AvgOfChans=np.mean(stimulus_code_rsquared)
#    stimulus_code_avg_rsquared_log=np.log(stimulus_code_rsquared_concatenated)
# 
#                   
#                                                
#    
#    trial_x[i]=stimulus_code_concatenatedBestChans
#    temp=trial_x[i]
#    #            ax2=plt.subplot(2,4,i+1)
#    #            ax2.plot(temp)
#    #            ax2.set_xlim(0,257)
#    #            ax2.set_ylim(-1,1)         
#    #           
#
#    oddball=oddball_stim[b,0][r,0] #the stimcode starts from 1 here. to get the event stim, add 7. e.g. 1--> 8: cow
#    if i+1==oddball:
#        trial_y[i]= 1
#
#
##for i in range (8):
##    stimulus_code_str=str(i+8)
##    trial_Epoch[stimulus_code_str].plot_image(2, cmap='interactive')
##    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
##    stimulus_code_avg.plot()
#
##ploting individual oddball epochs
#  
#oddball_code_str=str(oddball+7)   
#oddball_Epochs_data= trial_Epoch[oddball_code_str].get_data()
#oddball_Epochs_data_avg=np.mean(oddball_Epochs_data, axis=0)
#oddball_Epochs_data_avgOfChans=np.mean(oddball_Epochs_data, axis=1)
#oddball_Epochs_data_avgFz=oddball_Epochs_data_avg[2]
#oddball_Epochs_derivative=np.diff(oddball_Epochs_data)
#z=np.zeros((oddball_Epochs_data.shape[-1]))
#plt.figure(0)
#plt.plot(oddball_Epochs_data_avgFz)
#for i in range(oddball_Epochs_data.shape[0]):
#    plt.figure(1)
#    axavg=plt.subplot(1,oddball_Epochs_data.shape[0], i+1)
#    axavg.plot(10000*oddball_Epochs_data_avgOfChans[i])
#    axavg.plot(z)
#    axavg.set_ylim(-.5,.5)
#    
#    
#    oddball_Epoch_rsquared= np.square((oddball_Epochs_data[i][2]- np.mean(\
#                                          oddball_Epochs_data[i][2]\
#                                          )))/(float(np.size(oddball_Epochs_data[i][2]))*\
#                                            np.var(oddball_Epochs_data[i][2]))
#    ax4=plt.subplot(abs(oddball_Epochs_data.shape[0]/3)+1, 3, i+1)
#    ax4.plot(oddball_Epoch_rsquared)
#    ax4.set_ylim(-.1,.1)
#
#
#    
#        
#  
#    




#%% preparing the data for testing the classifier 
#b_test=3
#r_test=5
#test=AudioTactile_runs[b_test][r_test]
#test.n_times
#test.set_montage(montage)
#test_Epoch=mne.Epochs(test, exported_events, tmin=-.1, tmax=.55, reject=dict(eeg=8e-5), baseline=(-0.1,0)) #TODO: make tmin and tmax a variable
##        trial_Epoch.plot()
#test.filter(.3,20, method='iir') #4th order butterworth
#test.resample(sfreq=256)
#test_Epoch.load_data()
#test_Epoch.drop_channels(test_Epoch.info['bads'])
##ar = LocalAutoRejectCV()
##epochs_clean = ar.fit_transform(test_Epoch)
##        trial_Epoch.plot_image(8, cmap='interactive')
##trial_Epoch.drop_bad() # drops bad epochs automatically
#   
########creat feature vector for all stimuli in one run(trial) ########        
###trial_Epoch.drop_bad() 
#test_x=np.zeros( (8, numOfBestChans*test_Epoch.times.shape[0] ), dtype=float)
#test_x_avg=np.zeros( (8, test_Epoch.times.shape[0] ), dtype=float)
#test_y=np.zeros((8,1))
#t=(np.arange(0,168)-25.84)*3.87 # 168 data points from -100 s to 550 s to be 
##as time points in x-axis
#for i in range(8) :
#    stimulus_code_str=str(i+8)
##    test_Epoch[stimulus_code_str].plot_image(8, cmap='interactive')
#    stimulus_code_avg=test_Epoch[stimulus_code_str].average()
##    stimulus_code_avg.plot()
#
#    stimulus_code_avg_data = stimulus_code_avg.data
#
##normalizing the channels' data
#    stimulus_code_avg_data_norm=np.zeros((16, 168)) #TODO: make dimensions variable
#    for j in range(16):
#        stimulus_code_avg_data_norm[j]=\
#            stimulus_code_avg_data[j]/max(\
#                abs(stimulus_code_avg_data[j])) #normalization
#        plt.figure(j+1)
#        plt.subplot(2,4,i+1)
#        plt.plot(t,stimulus_code_avg_data_norm[j]) 
#        plt.axis([-100,550,-1,1])
#        
#    
##concatenating all channells after each other to be fed to the classifier as 
##the feature vector    
##    stimulus_code_concatenatedBestChans=stimulus_code_avg_data[bestChans].reshape(
##        np.size(stimulus_code_avg_data[bestChans]))
##    stimulus_code_concatenatedBestChans_norm= \
##        stimulus_code_concatenatedBestChans/max(\
##            abs(stimulus_code_concatenatedBestChans)) #normalization
##    
##    test_x_concatenated[i]=stimulus_code_concatenatedBestChans_norm
##          
###using the average of bestChans EEG data as the feature vector
##    stimulus_code_avgBestChans=stimulus_code_avg_data[bestChans].mean(axis=0)
##    stimulus_code_avgBestChans_norm= \
##        stimulus_code_avgBestChans/max(\
##            abs(stimulus_code_avgBestChans)) #normalization
##    
##    test_x_avg[i]=stimulus_code_avgBestChans_norm
##    plt.subplot(2,4,i+1)         
##    plt.plot(t,test_x_avg[i]) 
##    plt.axis([-100,550,-1,1])
#
#  
#    
#    oddball=oddball_stim[b_test,0][r_test,0] #the stimcode starts from 1 here. to get the event stim, add 7. e.g. 1--> 8: cow
#    if i+1==oddball:
#        test_y[i]= 1
#
#
##downsampling test_x with resamp_freq=~25HZ
##test_x_concatenated=test_x_concatenated[:,::10]
##test_x_avg=test_x_avg[:,::10]



#%% add SSP projections to 
#projs, events = mne.preprocessing.compute_proj_eog(trial_Epoch, n_grad=1, n_mag=1, average=True)
#print(projs)
#eog_projs = projs[-2:]
#mne.viz.plot_projs_topomap(eog_projs)

#%% Independent component analysis

#from mne.preprocessing import ICA
#n_components = 16  # if float, select n_components by explained variance of PCA
#method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
#decim = 3  # we need sufficient statistics, not all time points -> saves time
#
## we will also set state of the random number generator - ICA is a
## non-deterministic algorithm, but we want to have the same decomposition
## and the same order of components each time this tutorial is run
#random_state = 23
#ica = ICA(n_components=n_components, method=method, random_state=random_state)
#print(ica)
##reject = dict(mag=5e-12, grad=4000e-13)
#ica.fit(trial, decim=decim)
#print(ica)    
#
#ica.plot_components()
#ica.plot_properties(trial,picks=1,psd_args={'fmax': 35.})

#%% Feature extraction

#one componebt of the feature vectore could be the eeg data itself




