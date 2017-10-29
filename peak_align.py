# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 00:08:39 2017

@author: Amarantine
"""

#average over  channels and return peak-aligned stimulus epochs
            stim_Epochs_data_avgOfChans=np.mean(
                    trial_Epoch[stimulus_code_str].get_data(), axis=1)
            
            stim_Epochs_data_aligned=peak_align(stim_Epochs_data_avgOfChans,\
                                                trial_Epoch_long, tmin, tmax, decim,\
                                                stimulus_code_str)
            
            stim_Epochs_aligned_avgOfEpochs=np.mean(stim_Epochs_data_aligned, axis=0)
            stim_Epochs_aligned_avgOfEpochsChans=np.mean(stim_Epochs_aligned_avgOfEpochs, axis=0)
            minOfavgOfEpochsChans=np.min(stim_Epochs_aligned_avgOfEpochsChans)
            
#%% peak alignment
    
#avg channels for each epoch, choose the highest extrema (peak pick), (keep latncy magnitude)
#align all picks and everage all epochs for each stimulus code

def peak_align(stim_Epochs_data_avgOfChans, trial_Epoch_long,tmin, tmax, decim, stimulus_code_str):  
    """first input is a matrix containing average of best chans for the epochs of 
    a specific stimulus code. Its 0th dimention is number of stimulus code epochs,
        its 1st dimention is the length of each epoch. Peak_align returns the aligned 
        epoch. It means the peak of each epoch is shiftet to 300 miliseconds after onset."""
    trial_Epoch_long.drop_channels(trial_Epoch_long.info['bads'])
    stim_Epochs_data_aligned=np.zeros((trial_Epoch_long[stimulus_code_str].get_data().shape[0],
                                       trial_Epoch_long[stimulus_code_str].get_data().shape[1],
                                       np.int(1+(1000*(tmax-tmin)/decim))   ))
    
        
    for i in range(stim_Epochs_data_avgOfChans.shape[0]):
        peak_index=np.argmin(stim_Epochs_data_avgOfChans[i])
        
        peak_index=((peak_index*decim)+(tmin*1000)) #in seconds
        delta_latency=(peak_index - 300)/float(1000) #difference of peak latancy with 300 (p300)
        trial_Epoch_long_temp=trial_Epoch_long.copy()
        trial_Epoch_long_temp.drop_channels(trial_Epoch_long_temp.info['bads'])
        trial_Epoch_long_temp.crop(tmin=tmin + delta_latency, tmax=tmax+delta_latency)
        stim_Epochs_data_aligned_data=trial_Epoch_long_temp[stimulus_code_str].get_data()
        stim_Epochs_data_aligned[i]=stim_Epochs_data_aligned_data[i]
    return stim_Epochs_data_aligned
            