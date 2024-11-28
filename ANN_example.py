# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:03:36 2024

@author: haofg
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from joblib import load
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras import Input
from sklearn.metrics import r2_score,mean_absolute_error
import time
import sys

col_names = ['lon', 'lat', 'lai_hv', 'wind',
          'temperature', 'rain', 'pressure',
          'radiation', 'biomass']
mydir = './good_data/'
lai = np.load(mydir+"lai_hv_interp_congo_025.npy")
wind = np.load(mydir+"wind_interp_congo_025.npy")
temp = np.load(mydir+"temperature_interp_congo_025.npy")
rain = np.load(mydir+"rain_interp_congo_025.npy")
pres = np.load(mydir+"pressure_interp_congo_025.npy")
radi = np.load(mydir+"radiation_interp_congo_025.npy")
lon = np.load(mydir+"gedi_lon_congo_025.npy")
lat = np.load(mydir+"gedi_lat_congo_025.npy")
y = np.load(mydir+"gedi_agbd_congo_025.npy")

alldata = np.concatenate((lat.reshape(-1,1),lon.reshape(-1,1),lai.reshape(-1,1),\
                    wind.reshape(-1,1),temp.reshape(-1,1),rain.reshape(-1,1),\
                    pres.reshape(-1,1),radi.reshape(-1,1),y.reshape(-1,1)),axis=1)
alldata = alldata[~np.isnan(alldata).any(axis=1)] # remove nan
alldata = pd.DataFrame(alldata, columns=col_names)
print(alldata.shape)
pd.DataFrame(alldata).to_csv(mydir+'all_data.csv',index=False)
sys.exit()

X = alldata[:,:-1] # split into X
y = alldata[:,-1] # split into y

sc_X = MinMaxScaler(feature_range=(0,1))
sc_y = MinMaxScaler(feature_range=(0,1))

fitted_X_scaled = sc_X.fit_transform(X)
fitted_y_scaled = sc_y.fit_transform(y.reshape(-1,1))

# stratified random sampling
n_strata = 20 # cover 0-99% percentiles, the rest is assigned to the last strata bin
strata_bins = np.full([n_strata],np.nan)
for j in range(n_strata):
    strata_bins[j] = np.percentile(fitted_y_scaled,99/n_strata*(j+1))
strata_y = np.full([len(fitted_y_scaled),1],0)
for j in range(len(fitted_y_scaled)):
    if fitted_y_scaled[j] <= strata_bins[0]:
        strata_y[j] = 1
    elif fitted_y_scaled[j] <= strata_bins[n_strata-1]:
        for k in range(n_strata-1):
            if fitted_y_scaled[j] <= strata_bins[k+1] and fitted_y_scaled[j] > strata_bins[k]:
                strata_y[j] = k+1
    else:
        strata_y[j] = n_strata-1

X_train,X_test,y_train,y_test = train_test_split(fitted_X_scaled,fitted_y_scaled,test_size=0.2,stratify=strata_y,random_state=0)

# Build model
layers = [8, 10, 10, 10, 10, 1]
I = layers[0]
J = layers[-1]
nodes = layers[1:-1]
input_part1 = Input(shape=(I,))
ann_part1 = Dense(units=nodes[0], activation='softplus')(input_part1)
ann_part1 = Dense(units=nodes[1], activation='softplus')(ann_part1)
ann_part1 = Dense(units=nodes[2], activation='softplus')(ann_part1)
ann_part1 = Dense(units=nodes[3], activation='softplus')(ann_part1)
ann_out = Dense(units=1, activation='softplus')(ann_part1)
model = tf.keras.Model(inputs=input_part1, outputs=ann_out)
model.summary()

myepochs = 1000
mybatch_size = 10
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer = opt,loss = 'mse')

history = model.fit(X_train,y_train,\
              batch_size=mybatch_size,epochs=myepochs,\
              validation_data=(X_test,y_test))

plt.scatter(model.predict(X_train),y_train,label='Train')
plt.scatter(model.predict(X_test),y_test,label='Test')
plt.plot(range(0,2),range(0,2))
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(r2_score(y_train,model.predict(X_train)))
print(r2_score(y_test,model.predict(X_test)))

'''
save model tf
tf.keras.load_model
sc_x.transform() NOT FOIT TRANSFORM
dont rebuild model 
model.layer[0].trainable = false and so on
last 2-3 layers
'''