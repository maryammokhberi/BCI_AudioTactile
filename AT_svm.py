# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:06:08 2017

@author: mokhberim
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm

C=1
AT_svc=svm.SVC(kernel='linear', C=C).fit(x_balanced, y_balanced)

AT_rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_balanced, y_balanced)

AT_poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_balanced, y_balanced)

AT_lin_svc = svm.LinearSVC(C=C).fit(x_balanced, y_balanced)
