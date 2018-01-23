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
chance=55
acceptable=70
#no_meditation_means = [61,86,61,79,76,61,64,54,70,72,51,90]
#no_meditation_std = [11,9,13,10,12,12,15,11,14,12,8,10]
meditationVsRest_means=[82,78,99,98,100,80,68,64,93,93,63,89]
meditationVsRest_std=[3,10,1,1,0,1,10,6,5,7,15,06]


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
#color=np.random.rand(3,)
rects1 = ax.bar(ind, meditationVsRest_means, width, color='k', yerr=meditationVsRest_std)

#meditation_means = (60,71,68,75,77,52,58,68,80,73,70,97)
#meditation_std = (12,13,13,12,11,13,14,16,12,12,15,1)
#rects2 = ax.bar(ind + width, meditation_means, width, color='mediumturquoise', yerr=meditation_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('AUC')
ax.set_title('Classification of Meditaion versus Rest Condition')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('S1', 'S2', 'S3', 'S4', 'S5','S6', 'S7', 'S8', 'S9', 'S10','S11','S12'))
#ax.plot([-1, 12], [chance, chance], "k--")
#ax.plot([-1, 12], [acceptable, acceptable], "k--")
#ax.legend((rects1[0], rects2[0]), ('without meditation', 'with meditation'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
#autolabel(rects2)

plt.show()