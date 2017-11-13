# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:24:21 2017

@author: Amarantine
"""
import pywt

x_wavelet=trial_Epoch_data[37,1,:]


from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(0, 0.8, 200, endpoint=False)
widths = np.arange(0.1, 31)
cwtmatr = signal.cwt(x_wavelet, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[0, 0.8, 1, 31], cmap='PRGn', aspect='auto',
vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()