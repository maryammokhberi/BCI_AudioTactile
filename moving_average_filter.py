# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 22:00:14 2017

@author: Amarantine
"""

import numpy as np

def MovAvgFilt(InMatrix, Len, Param):
#% The moving average filter operates by averaging a number of points from the
#% input signal to produce each point in the output signal.  In equation form,
#% this is written:
#%
#%         1  M-1
#% Y[i] = --- SUM X[i + j]
#%         M  j=0
#% Example Usage:
#% t = 1:1/100:2;
#% x1 = sin (2*pi*5*t);
#% x2 = rand(1,101);
#% In = x1 + x2;
#% Param = 'Center';
#% Len = 5;
#% Out = func_MovAvgFilt (In, Len, Param);
#% plot (t, x1); title ('Original Signal')
#% figure
#% plot (t, In); title ('Noisy Signal')
#% figure
#% plot (t, Out); title ('Filtered Signal')
    OutMatrix=np.zeros(InMatrix.shape)
    Out =np.zeros((InMatrix.shape[1]))
    for NumOfRows in range(InMatrix.shape[0]):
        In=InMatrix[NumOfRows,:]

        Siz_In=In.shape[0]
        
        if (Param== 'Left'):
            Pad = np.zeros ((1, Len - 1))
            New_In = np.array([Pad, In])
            for i in range (Siz_In):
                temp = 0
                for j in range (Len):
                    temp = temp + New_In[i + j - 1]
                
                Out[i] = temp / float(Len)
            
            
        elif (Param=='Center'):
            len1 = Len%2
            if (len1== 0):
                raise NameError('Cannot use the Len as an even number for this option. Use Left or Right')
            else:
                Pad_Len = (Len - 1)/2
                Pad = np.zeros ((Pad_Len))
                New_In = np.concatenate([Pad, In, Pad],axis=0)
                for i in range (Siz_In):
                    temp = 0
                    for j in range (Len):
                        temp = temp + New_In[i + j - 1]
                    
                    Out[i] = temp / Len
                
            
            
        elif (Param== 'Right'):
            Pad = np.zeros ((1, Len - 1))
            New_In = np.array([In, Pad])
            for i in range (Siz_In):
                temp = 0
                for j in range (Len):
                    temp = temp + New_In[i + j - 1]
                
                Out[i] = temp / Len
            
        
        OutMatrix[NumOfRows,:]=Out
    
    
    return  OutMatrix