# -*- coding: utf-8 -*-
"""
@author:mokhberim
"""
#import necessary files

import os
import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#change directory
os.chdir('T:\Data\data set 2\BCI_Comp_III_Wads_2004\BCI_Comp_III_Wads_2004')

#load data
A_train=sio.loadmat('Subject_A_Train.mat')
Signal= A_train['Signal']
Flashing=A_train['Flashing']
StimulusType= A_train['StimulusType']
#t=np.arange(0,32.47500,0.00416666666)
#t=t[0:7794]

org_sig=np.zeros([12,144,15,64,85], dtype=float)
labels_all=np.zeros([85,12,15], dtype=float)

for t in range(85):
    org_tar_sig=np.zeros([12,144,15,64], dtype=float)
    for ch in range(64):
    
        Signal_chan=Signal[:,:,ch] #first target signal, each channel
        sig_len=len(Signal[t,:,ch]) #length of the time dimention of the Signal
        StimulusCode=A_train['StimulusCode']
        org_tar_sig_chan=np.zeros([12,144,15], dtype=float) #600 ms after the onset of stimulus
        code_flash_num=np.zeros([12], dtype=int)
        code_index=0
        #make a 3D array whith axis0=different codes(rows/columns), axis1=signal for each flash
        #and axis2= 15 repeatitions
                
        for i in range(sig_len-1): 
            if (Flashing[t,i] - Flashing[t, i+1])==1 :
                code_index=int(StimulusCode[t,i])
                code_flash_num[code_index-1]+=1 
                org_tar_sig_chan[code_index-1, :, (code_flash_num[code_index-1])-1]= Signal_chan[t,i-23:i+121] 
                labels_all[t, code_index-1, (code_flash_num[code_index-1])-1 ]= StimulusType[t, i] #prepare labels from StimulusType matrix
        #sum over the 15 repeatitions of data points for one channel
        org_sig_sum_chan=np.sum(org_tar_sig, axis=2)
        
        
        #put all 64 channel together 
        org_tar_sig[:,:,:,ch]=org_tar_sig_chan
        org_sig_sum=np.sum(org_tar_sig, axis=2)
        
    #put all targets and channels together; a 5D array: rows/columns, data points, 15 repeatitions, channels, targets
    org_sig[:,:,:,:,t]=org_tar_sig
    
accumulated_org_sig=np.sum(org_sig, axis=2)
labels= np.mean(labels_all, axis=2) #contains lables for accumulated_org_sig
#%%compute X and Y to be fed into a classifier
X=np.zeros([1020,144,64]) # 85*12=1020
for i in range(accumulated_org_sig.shape[3]):
    X[12*i:12*i+12,:,:]= accumulated_org_sig[:,:,:,i]
np.save('X', X)    
               
Y=np.zeros([1020])
for i in range(labels.shape[1]) :
    Y[12*i:12*i+12]= labels[i,:]
 
np.save('Y', Y)        
               
#%%
#Cz=accumulated_org_sig[:,:,10,:]
#CzA=Cz[:,:,1]           
#lettermapA=[np.mean((CzA[i,:],CzA[j,:]),axis=0) for i in [0,1,2,3,4,5] for j in [6,7,8,9,10,11]]               
#for i in range(36):    
#  plt.subplot(6,6,(i+1))    
#  plt.plot(lettermapA[i])
#  plt.ylim([-100,120])
#  plt.xlim([50,100])               
#    
#  
  
  
#%%
    #plot the graphs corresponding all flashes
#    for i in range(12):    
#        plt.subplot(2,6,(i+1))    
#        plt.plot(org_sig_sum[i,:,CHANNEL]) 
#        plt.ylim(-100,160)

#%%targetchar

#TargetChar= A_train['TargetChar']




#%%
#from sklearn.decomposition import PCA
#
#for code in range(12):
#    X=accumulated_org_sig[code,:,:,0]
#    pca=PCA(n_components=1)
#    pca.fit(X)
#    Xpca=pca.transform(X)
#    accumulated_org_sig_pca1[code]=Xpca[:,0]
#
#
#
#for i in range(12):    
#  plt.subplot(3,4,(i+1))          
#  plt.plot(accumulated_org_sig_pca1[i])

#%%  R-squared: proportion of variance for each datapoint 
#for the first target in Cz

#tar1Cz=accumulated_org_sig[:,:,10,0]
#
#
#def rSq(sig):
#    varsig=np.var(sig) 
#    lensig=len(sig)
#    rSquared=np.zeros(lensig)
#    rSquared= [((sig[i]- np.mean(sig))**2)/(varsig*lensig) for i in range(lensig) ]
#    return rSquared
#tar1Cz_rSq=np.zeros([12,144])
#for i in range(12):
#    tar1Cz_rSq[i]= rSq(tar1Cz[i,:])
#for i in range(12):
#    plt.subplot(2,6,i+1)
#    plt.plot(tar1Cz_rSq[i])
#%% adding events to the 
     
