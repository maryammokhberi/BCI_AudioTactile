# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:24:21 2017

@author: Amarantine
"""
import pywt
import random
from scipy import signal
import matplotlib.pyplot as plt

plt.close('all')

cmap=random.choice(plt.colormaps())
plt.figure(1)
x_wavelet=baseline_Epoch_data[23,8,:]

t = np.linspace(0, 60, 1, endpoint=False)
widths = np.arange(2, 20)
cwtmatr = signal.cwt(x_wavelet, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[0, 60, 1, 20],cmap=cmap, aspect='auto',
vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

plt.figure(2)
x_wavelet_meditation2=meditation2_Epoch_data[23,8,:]
t = np.linspace(0, 1, 60, endpoint=False)
widths = np.arange(2, 20)
cwtmatr = signal.cwt(x_wavelet_meditation2, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[0, 60, 1, 20], cmap=cmap, aspect='auto',
vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

#%%
plt.figure(3)
freqs = np.arange(7,30, 1)
n_cycles=2
from mne.time_frequency import tfr_morlet  # noqa
power, itc = tfr_morlet(baseline_Epoch, freqs=freqs, n_cycles=n_cycles,
                        return_itc=True, decim=3, n_jobs=1)
power.plot([power.ch_names.index('Cz')])

plt.figure(4)
freqs = np.arange(7,30, 1)
n_cycles=2
from mne.time_frequency import tfr_morlet  # noqa
power, itc = tfr_morlet(meditation2_Epoch, freqs=freqs, n_cycles=n_cycles,
                        return_itc=True, decim=3, n_jobs=1)
power.plot([power.ch_names.index('Cz')])


#%%
#cmap=random.choice(plt.colormaps())
#cmap='BrBG_r' #'Spectral_r', cubehelix_r, Pastel1,RdYlBu_r, YlGnBu_r,PuBuGn, summer_r, bwr_r, pink_r, ocean, Set2
cmap='Spectral_r'
channelNum=7
ax1 = plt.subplot(121)
Pxx, freqs, bins, im = plt.specgram(baseline.get_data()[channelNum,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap)
plt.xlabel('baseline')  
plt.ylim(0,20)
plt.colorbar()
plt.show()

plt.subplot(122, sharex=ax1)
Pxx, freqs, bins, im = plt.specgram(meditation2.get_data()[channelNum,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap)
plt.xlabel('meditation2')
plt.ylim(0,20)
plt.colorbar()
plt.show()
