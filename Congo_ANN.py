# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:39:18 2024

@author: haofg
ANN on congo data
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math

#load data
inputs = ['gedi_lon_congo_025','gedi_lat_congo_025','lai_hv_interp_congo_025',
          'temperature_interp_congo_025','rain_interp_congo_025','radiation_interp_congo_025',
          'pressure_interp_congo_025','wind_interp_congo_025']
col_names = ['lon','lat','lai_hv',
          'temperature','rain','radiation',
          'pressure','wind','biomass']  
mydir = '../../GEDI-biomass_all_models/'
datadir = './good_data/'
X = np.empty([3657]) # remeber to change

for i in inputs:
    f = np.load(datadir + i + '.npy', 'r')
    copy = f.copy()
    copy[copy <= 0] = np.nan
    X = np.vstack((X, copy))

y = np.load(datadir + 'gedi_agbd_congo_025.npy', 'r')
y = np.array(y)
#sys.exit()
#y[y <= 50] = np.nan # cahnged from y < 0
  
X = np.vstack((X,y))
inputs.append('y')
print(X.shape)
X = pd.DataFrame(X[1:, :].T, columns=col_names)
X = X.dropna()
print(X.shape)
#pd.DataFrame(X).to_csv(mydir+'all_data.csv',index=False)

#graph PDF hists
    
for i in range(2, len(inputs)-1):
    plt.subplot(2,3,i-1)
    plt.hist(X[col_names[i]], bins = 100, density = True)
    plt.title(inputs[i])
plt.tight_layout()
plt.show()
plt.hist(X['biomass'], bins=100, density = True)
plt.title('biomass distr')
plt.show()

sys.exit()
#define input/output
y = X['y']
X = X[inputs[:-1]]
#train test split
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.2, random_state=0) #tf is randomstate
'''
model_id = 8
X_train = pd.read_csv(mydir+'model'+str(model_id+1)+'/Xtrain.csv').values
X_test = pd.read_csv(mydir+'model'+str(model_id+1)+'/Xtest.csv').values
y_train = pd.read_csv(mydir+'model'+str(model_id+1)+'/ytrain.csv').values
y_test = pd.read_csv(mydir+'model'+str(model_id+1)+'/ytest.csv').values
# ANN
'''
# Adding the input layer and the first hidden layer
nodes = 32
drop = 0.1
input_dim = 8
ann = Sequential([
    Input(input_dim),
    Dense(nodes*4, activation = 'relu'),
    #Dropout(drop),
    Dense(nodes*8, activation = 'relu'),
    #Dropout(drop),
    Dense(nodes*4, activation = 'relu'),
    #Dropout(drop),
    Dense(nodes*2, activation = 'relu'),
    #Dropout(drop),
    Dense(nodes*1, activation = 'relu'),
    #Dropout(drop),
    Dense(1, activation = 'linear'),
    ])

features = 8
input_part1 = Input(shape=(features,))
layers=[26,25,19,25,16,25,17,21,30,24,25,20,23,26]
for n_layers in layers:
    if n_layers == layers[0]:
        ann_part1 = Dense(units=n_layers, activation='softplus')(input_part1)
    else:
        ann_part1 = Dense(units=n_layers, activation='softplus')(ann_part1)

out = Dense(units=1, activation='linear')(ann_part1)
ann = tf.keras.Model(inputs=input_part1, outputs=out)

print(ann.summary())

rmse = tf.metrics.RootMeanSquaredError()
r2_metric = tf.keras.metrics.R2Score()

kwargs = {}
kwargs["learning_rate"] = 0.0001204052358523759
opt = getattr(tf.optimizers, 'Adam')(**kwargs)
ann.compile(optimizer = opt, loss = "mse", metrics = [rmse, r2_metric])

epochs = 1000
history = ann.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_data=(X_test, y_test))

ann.save('initial_ann.keras')
model = load_model('initial_ann.keras')
y_pred = model.predict(X_test)
plt.scatter(y_pred, y_test, s=2)
plt.show()

# printing r2 and mse and rmse
print(mean_squared_error(y_pred, y_test))
print(math.sqrt(mean_squared_error(y_pred, y_test)))
print(r2_score(y_pred, y_test))

#epochs = 100
#history = ann.fit(X_train, y_train, batch_size=100, epochs=epochs, validation_data=(X_test, y_test))
train_accuracy = history.history['loss']
val_accuracy = history.history['val_loss']
plt.plot(range(2,epochs+1), train_accuracy[1:], color='blue', lw=3, label='training')
plt.plot(range(1,epochs+1), val_accuracy, color='red', lw=3, label='validation')
plt.xlim(0, 1500)
plt.ylim(0,1000)
plt.xlabel('epochs')
plt.ylabel('mse')
plt.show()

train_accuracy = history.history['r2_score']
val_accuracy = history.history['val_r2_score']
plt.plot(range(2,epochs+1), train_accuracy[1:], color='blue', lw=3, label='training')
plt.plot(range(1,epochs+1), val_accuracy, color='red', lw=3, label='validation')
plt.xlim(0,1500)
plt.ylim(-10,10)
plt.xlabel('epochs')
plt.ylabel('r2')
plt.show()

#SCATTER HIST JOINT PLOT
g = sns.jointplot(y = y_pred.reshape(-1,), x = y_test.reshape(-1,), kind = 'reg')
g.set_axis_labels('AGBD', 'model predictions')



