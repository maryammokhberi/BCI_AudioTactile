# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:41:30 2018

@author: Amarantine
"""

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

N = 6
trialNum_mean=[8, 1,12,12,8.33,8.4 ]
trialNum_std=[0, 0,0,0,5.18,4.4]


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
color=np.random.rand(3,)
rects1 = ax.bar(ind, trialNum_mean, width, color=color, yerr=trialNum_std)

#meditation_means = (60,71,68,75,77,52,58,68,80,73,70,97)
#meditation_std = (12,13,13,12,11,13,14,16,12,12,15,1)
#rects2 = ax.bar(ind + width, meditation_means, width, color='mediumturquoise', yerr=meditation_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Trial number')
ax.set_title('Age')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('10', '11', '14', '15', '16', '17',))
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