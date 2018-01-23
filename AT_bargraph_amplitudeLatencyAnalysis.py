# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 09:06:46 2018

@author: Amarantine
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
N = 12


#word_latency_meanOfAllParticipants=[0.498, 0.495, 0.507, 0.497, 0.510, 0.493,0.527,0.493] #generated in AT_latencyAmplitudeStatistics
#word_latency_stdOfAllParticipants=[0.022,0.017,0.026,0.022,0.017,0.019,0.027,0.022]
#speaker_latency_meanOfAllParticipants=speaker_latency_meanOfAllParticipants
#speaker_latency_stdOfAllParticipants=speaker_latency_stdOfAllParticipants


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
#color_latency=np.array([ 0.88529234,  0.75359132,  0.18130554])
rects1 = ax.bar(ind, tempnomedmean, width, color='lightcoral', yerr=tempnomedstd)

#meditation_means = (60,71,68,75,77,52,58,68,80,73,70,97)
#meditation_std = (12,13,13,12,11,13,14,16,12,12,15,1)
rects2 = ax.bar(ind + width, tempmean, width, color='mediumturquoise', yerr=tempstd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Time (seconds)')
ax.set_title('Comparison of P300 latency in control and study sessions')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12'))
ax.legend((rects1[0], rects2[0]), ('without meditation', 'with meditation'))


#def autolabel(rects):
#    """
#    Attach a text label above each bar displaying its height
#    """
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                '%d' % int(height),
#                ha='center', va='bottom')
#
#autolabel(rects1)
#autolabel(rects2)

plt.show()