# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 14:04:00 2017

@author: mokhberim
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(x, y)