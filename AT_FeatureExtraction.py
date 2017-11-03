# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:42:00 2017

@author: Amarantine
"""
import numpy as np


#1: latency
def latency(inputEpoch, tmin, decim):
    """input: 1D input epoch,
       returns: the ERP’s latency time, i.e. the time
       where the maximum signal value appears in seconds"""
    inputEpoch_maxpoint=np.argmax(inputEpoch)+1
    latency= (inputEpoch_maxpoint*decim)/float(1000) + tmin
    return latency


#2: Amplitude
def amplitude(inputEpoch):
    """maximum signal value"""
    amplitude= np.max(inputEpoch)
    return amplitude

#3: Latency/amplitude ratio (LAR)
def lat_amp_ratio (inputEpoch, tmin, decim):
    """latency/amplitude ration"""
    inputEpoch_maxpoint=np.argmax(inputEpoch)+1
    latency= (inputEpoch_maxpoint*decim)/float(1000) + tmin
    amplitude= np.max(inputEpoch)
    lat_amp_ratio=latency/amplitude
    return lat_amp_ratio

#4: absolute amplitude
def abs_amp(inputEpoch):
    """absolute amplitude"""
    abs_amp=np.abs(np.max(inputEpoch))
    return abs_amp
 
#5: Absolute latency/amplitude ratio
def abs_lat_amp_ratio(inputEpoch, tmin, decim):
    """Absolute latency/amplitude ratio"""
    inputEpoch_maxpoint=np.argmax(inputEpoch)+1
    latency= (inputEpoch_maxpoint*decim)/float(1000) + tmin
    amplitude= np.max(inputEpoch)
    abs_lat_amp_ratio=np.abs(latency/amplitude)
    return abs_lat_amp_ratio

#7: positive area (the sum of the positive signal values)
def positive_area(inputEpoch):
    """—the sum of the positive signal values"""
    positive_area=np.sum(inputEpoch[inputEpoch>0])
    return positive_area

#7: negative area (—the sum of the negative signal values)
def negative_area(inputEpoch):
    """—the sum of the negative signal values"""
    negative_area=np.sum(inputEpoch[inputEpoch<0])
    return negative_area

#8: total area
def total_area(inputEpoch):
    """sum of all data points"""
    total_area=np.sum(inputEpoch)
    return total_area
#9: abs_total_area 
def abs_total_area(inputEpoch):
    """abs_total_are"""
    abs_total_area=np.abs(np.sum(inputEpoch))
    return abs_total_area    

#10: total basolute area
def total_abs_area(inputEpoch):
    """ sum of positive area and abs negative area"""
    total_abs_area=np.sum(inputEpoch[inputEpoch>0]) +\
                   np.abs(np.sum(inputEpoch[inputEpoch<0]))
    return total_abs_area

#11: Average absolute signal slope
def avg_abs_slope(inputEpoch):
    """average of slopes (epoch derivative)"""
    avg_abs_slope=np.mean(np.abs(np.diff(inputEpoch)))
    return avg_abs_slope

#12: peak_to_peak
def peak_to_peak(inputEpoch):
    """difference between maximum and minimum value of the signal"""
    peak_to_peak=np.max(inputEpoch) - np.min(inputEpoch)
    return peak_to_peak

#13: peak_to_peak time window
def pk_to_pk_tw(inputEpoch,decim):
    """peak to peak time window: time diffrence of max point and min point
       in the epoch in seconds"""
    pk_to_pk_tw=(np.argmax(inputEpoch)-np.argmin(inputEpoch))*decim/float(1000)
    return pk_to_pk_tw

#14: peak_to_peak slope
def pk_to_pk_slope(inputEpoch,decim):
    """peak_to_peak/peak_to_peak time winndow"""
    peak_to_peak=np.max(inputEpoch) - np.min(inputEpoch)
    pk_to_pk_tw=(np.argmax(inputEpoch)-np.argmin(inputEpoch))*decim/float(1000)
    pk_to_pk_slope= peak_to_peak/pk_to_pk_tw
    return pk_to_pk_slope

#15 Zero crossings 
def zero_cross(inputEpoch):
    """the number of times t that s(t)=0, in peak-to-peak time window"""
    maxmin=np.array((np.argmax(inputEpoch),np.argmin(inputEpoch))) 
    zero_crossing= np.size(np.where(np.diff(np.sign(inputEpoch[
            np.min(maxmin):np.max(maxmin)]))))
    return zero_crossing

#16: zero crossing density
def zero_cross_density(inputEpoch,decim):
    """zero crossings per time unit, in peak-to-peak time window"""
    maxmin=np.array((np.argmax(inputEpoch),np.argmin(inputEpoch))) 
    zero_crossing= np.size(np.where(np.diff(np.sign(inputEpoch[
            np.min(maxmin):np.max(maxmin)]))))
    zero_cross_density=zero_crossing/((np.argmax(inputEpoch)-np.argmin(inputEpoch))
                                   *float(decim))
    return zero_cross_density

#17: slope sign alteration
def slope_sign_alt(inputEpoch):
    """the number of slope sign alterations"""
    inputEpoch_diff=np.diff(inputEpoch)
    slope_sign_alt= np.size(np.where(np.diff(np.sign(inputEpoch_diff))))
    return slope_sign_alt


    
    
    
    
    
    

    
     
    


          
    
       
    
    
        

     
    
    
    
    