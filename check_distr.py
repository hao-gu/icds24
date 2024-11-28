# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:12:40 2024

@author: haofg
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
#constants
xmin = 3
xmax = 30

ymin = -7
ymax = 7
#load data
inputs = ['lai_hv_interp_congo','temperature_interp_congo','rain_interp_congo','radiation_interp_congo',
          'pressure_interp_congo','wind_interp_congo']
inputs2 = ['lai_hv_interp_congo_025','temperature_interp_congo_025','rain_interp_congo_025','radiation_interp_congo_025',
          'pressure_interp_congo_025','wind_interp_congo_025']
grid = np.load('grid_new.npy','r')
grid_025 = np.load("grid_new_025.npy", 'r')
for i in range(len(inputs)):
    a = np.full((271, 141), np.nan)
    f = np.load(inputs[i] + '.npy', 'r')
    for point, data in zip(grid, f):
        x = (point[0] - xmin) * 10
        y = (point[1] - ymin) * 10
        a[int(x), int(y)] = data
    a2 = np.full((109, 57), np.nan)
    f = np.load(inputs2[i] + '.npy', 'r')
    for point, data in zip(grid, f):
        x = (point[0] - xmin) * 4
        y = (point[1] - ymin) * 4
        a2[int(x), int(y)] = data
    plt.imshow(a)
    plt.show()
    plt.imshow(a2)
    plt.show()
    
    #lon distr
    x = np.arange(xmin, xmax + 0.1, 0.1)
    x2 = np.arange(xmin, xmax + 0.25, 0.25)
    #lat distr
    y = np.arange(ymin, ymax + 0.1, 0.1)
    y2 = np.arange(ymin, ymax + 0.25, 0.25)
    
    #latitude distr 0.1
    plt.plot(y, np.nanmean(a, axis=0), label = '0.1')
    #latitude distr 0.25
    plt.plot(y2, np.nanmean(a2, axis=0), label = '0.25')
    plt.title(inputs[i] + " latitudinal distribution")
    plt.legend(loc = 'upper left')
    plt.show()
    
    #longitudinal distr 0.1
    plt.plot(x, np.nanmean(a, axis=1), label = '0.1')
    #longitudinal distr 0.25
    plt.plot(x2, np.nanmean(a2, axis=1), label = '0.25')
    plt.legend(loc = 'upper left')
    plt.title(inputs[i] + " longitudinal distrubition")
    plt.show()
    #data_copy.nanmean()
    #X = np.vstack((X, copy))
