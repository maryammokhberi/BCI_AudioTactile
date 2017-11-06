# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 17:30:12 2017

@author: Amarantine
"""
#sending sample data to Tom, par0130, b=0, r=0, epochs sorted based on stimulus code, oddball=7(14)
import numpy as np
import scipy.io as sio
temp=X_multi_D[0,0,:,:,0:16,:]
temp.shape
temp_reshape=temp.reshape((8*13,16,41))
temp_reshape.shape
nan_elements=np.argwhere(np.isnan(temp_reshape)==True)
nan_index=nan_elements[:,0]
nan_index=np.unique(nan_index)
temp_reshape_100=np.delete(temp_reshape, nan_index, axis=0)
temp_reshape_100.shape


labels=np.zeros((100,1))
labels[75:88]=np.ones((13,1))
tempdict=dict(run_data=temp_reshape_100, labels=labels)

sio.savemat('data_Epochs.mat', tempdict)
