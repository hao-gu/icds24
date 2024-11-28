# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:26:50 2024

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
import sys
from heatmap import heatmap, annotate_heatmap

#load data
inputs = ['gedi_lon_congo_025','gedi_lat_congo_025','lai_hv_interp_congo_025',
          'temperature_interp_congo_025','rain_interp_congo_025','radiation_interp_congo_025',
          'pressure_interp_congo_025','wind_interp_congo_025']
col_names = ['lon','lat','lai_hv',
          'temperature','rain','radiation',
          'pressure','wind','biomass']  
mydir = '../../../ICDS_paper/Graphs/'


X = np.empty([3657]) # remeber to change


for i in inputs:
    f = np.load(i + '.npy', 'r')
    copy = f.copy()
    copy[copy <= 0] = np.nan
    X = np.vstack((X, copy))

y = np.load('gedi_agbd_congo_025.npy', 'r')
y = np.array(y)
y[y <= 50] = np.nan # cahnged from y < 0
  
X = np.vstack((X,y))
inputs.append('y')

X = pd.DataFrame(X[1:, :].T, columns=col_names)
X = X.dropna()
#pd.DataFrame(X).to_csv(mydir+'all_data.csv',index=False)

#heatmap
#congo
temp = X.corr(method='pearson')
print(X.corr(method='kendall'))
print(X.corr(method='spearman'))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12,10))
im, cbar = heatmap(temp, col_names, col_names, ax=ax2,
                   cmap="YlGn", cbarlabel="pearsons correlation")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()

#amazon
amazondir = '../amazonCodes/'
inputs = ['gedi_lon','gedi_lat','lai_hv_interp','temperature_interp','rain_interp','radiation_interp','pressure_interp','wind_interp']
X = np.empty([42999]) # remeber to change
for i in inputs:
    f = np.load(amazondir + i + '_025.npy', 'r')
    X = np.vstack((X, f))

y = np.load(amazondir+'gedi_agbd_025.npy', 'r')
y = np.array(y)
y[y <= 0] = np.nan # cahnged from y < 0

X = np.vstack((X,y))
inputs.append('y')
X = pd.DataFrame(X[1:, :].T, columns=col_names)
X = X.dropna()

#variable correlation
temp = X.corr(method='pearson')
print(X.corr(method='kendall'))
print(X.corr(method='spearman'))

im, cbar = heatmap(temp, col_names, col_names, ax=ax,
                   cmap="YlGn", cbarlabel="pearsons correlation")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()

plt.savefig( mydir+'figure2.png', dpi=300)
plt.show()

#ax = sns.heatmap(temp, annot=True, linewidth = .5, cbar = True, fmt = '.1f')


sys.exit()