# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:15:49 2017

@author: Amarantine
"""
r=2
b=3
print "********preparing data for the stimuli in block", b+1,"run#", r+1,"****************\
        *********************"
trial=AudioTactile_runs[b][r]
trial.n_times

trial.set_montage(montage)
trial.filter(.3,20, method='iir') #4th order butterworth
trial.resample(sfreq=256)

## Default MNE referencing: average of all channels
#
#trial.set_montage(montage)
#trial.filter(.3,20, method='iir') #4th order butterworth
#trial.resample(sfreq=256)
#trial_Epoch=mne.Epochs(trial, exported_events, tmin=-.1, tmax=.55,\
#                       reject=dict(eeg=8e-5), baseline=(-0.1,0)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate
#
#title= 'referenced to the average of all channels'                       
#for i in range(2) :
#    stimulus_code_str=str(i+8)
#    #plot average of stim_code epochs voltage + cmap of individuals
#    trial_Epoch[stimulus_code_str].plot_image(8, cmap='interactive')
#    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
#    stimulus_code_avg.plot(titles=dict(eeg=title)) 





# Original refrencing (physical referencing)
trial_reReferenced, _ =mne.set_eeg_reference(trial, [])
trial.set_montage(montage)
trial.filter(.3,20, method='iir') #4th order butterworth
trial.resample(sfreq=256)
trial_Epoch=mne.Epochs(trial_reReferenced, exported_events, tmin=-.1, tmax=.55,\
                       reject=dict(eeg=8e-5), baseline=(-0.1,0)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate

title= 'EEG original reference'                                              
for i in range(2) :
    stimulus_code_str=str(i+8)
    #plot average of stim_code epochs voltage + cmap of individuals
    trial_Epoch[stimulus_code_str].plot_image(8, cmap='interactive')
    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
    stimulus_code_avg.plot(titles=dict(eeg=title)) 


    
#refrencing to a single channel

trial_reReferenced, _=mne.set_eeg_reference(trial, ['Fz'])
trial.set_montage(montage)
trial.filter(.3,20, method='iir') #4th order butterworth
trial.resample(sfreq=256)
trial_Epoch=mne.Epochs(trial_reReferenced, exported_events, tmin=-.1, tmax=.55,\
                       reject=dict(eeg=8e-5), baseline=(-0.1,0)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate

title= 'EEG referenced to Fz'                                              
for i in range(2) :
    stimulus_code_str=str(i+8)
    #plot average of stim_code epochs voltage + cmap of individuals
    trial_Epoch[stimulus_code_str].plot_image(8, cmap='interactive')
    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
    stimulus_code_avg.plot(titles=dict(eeg=title)) 


##refrencing to a mean of some channels

#trial_reReferenced, _=mne.set_eeg_reference(trial, ['T7', 'T8'])
#trial.set_montage(montage)
#trial.filter(.3,20, method='iir') #4th order butterworth
#trial.resample(sfreq=256)
#trial_Epoch=mne.Epochs(trial_reReferenced, exported_events, tmin=-.1, tmax=.55,\
#                       reject=dict(eeg=8e-5), baseline=(-0.1,0)) #TODO: make tmin and tmax a variable, make sure thredhold for eeg is appropriate
#
#title= 'EEG referenced to mean of T7 and T8'                                              
#for i in range(2) :
#    stimulus_code_str=str(i+8)
#    #plot average of stim_code epochs voltage + cmap of individuals
#    trial_Epoch[stimulus_code_str].plot_image(8, cmap='interactive')
#    stimulus_code_avg=trial_Epoch[stimulus_code_str].average()
#    stimulus_code_avg.plot(titles=dict(eeg=title)) 