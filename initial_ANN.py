# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:02:35 2023

@author: haofg
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
#load data
inputs = ['gedi_lon1','gedi_lat','lai_hv_interp','temperature_interp','rain_interp','radiation_interp','pressure_interp','wind_interp']
X = np.empty([10898]) # remeber to change
for i in inputs:
    f = np.load(i + '.npy', 'r')
    X = np.vstack((X, f))

y = np.load('gedi_agbd.npy', 'r')
y = np.array(y)
y[y <= 0] = np.nan # cahnged from y < 0

X = np.vstack((X,y))
inputs.append('y')
X = pd.DataFrame(X[1:, :].T, columns=inputs)
X = X.dropna()

#graph PDF hists
sns.set_theme()
for i in range(2, len(inputs)-1):
    plt.subplot(2,3,i-1)
    plt.hist(X[inputs[i]], bins = 100, density = True)
    plt.title(inputs[i])
plt.tight_layout()
plt.show()
plt.hist(X['y'], bins=100, density = False)
plt.title('biomass distr')
plt.show()
#variable correlation
temp = X.corr(method='pearson')
print(X.corr(method='kendall'))
print(X.corr(method='spearman'))
ax = sns.heatmap(temp, annot=True, linewidth = .5, cbar = True, fmt = '.1f')

#define input/output
y = X['y']
X = X[inputs[:-1]]
#train test split
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.2, random_state=0) #tf is randomstate
'''
# ANN
ann = Sequential()
# Adding the input layer and the first hidden layer
ann.add(Dense(units=10, input_dim = 8, activation='relu'))
# Adding the second hidden layer
ann.add(Dense(units=10, activation='relu'))
ann.add(Dense(units=10, activation='relu'))
ann.add(Dense(units=10, activation='relu'))
ann.add(Dense(units=10, activation='relu'))
# Adding the output layer
ann.add(Dense(units=1, activation='linear'))

print(ann.summary())

rmse = tf.metrics.RootMeanSquaredError()
r2_metric = tf.keras.metrics.R2Score()
ann.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [rmse, r2_metric])

epochs = 100
history = ann.fit(X_train, y_train, batch_size=100, epochs=epochs, validation_data=(X_test, y_test))

ann.save('initial_ann.h5')
model = load_model('initial_ann.h5')
y_pred = model.predict(X_test)
plt.scatter(y_pred, y_test, s=2)
plt.show()

# Training the ANN on the Training set, hyper-parameters, batch_size, epochs


epochs = 100
history = ann.fit(X_train, y_train, batch_size=100, epochs=epochs, validation_data=(X_test, y_test))
train_accuracy = history.history['rmse']
val_accuracy = history.history['val_rmse']
plt.plot(range(2,epochs+1), train_accuracy[1:], color='blue', lw=3, label='training')
plt.plot(range(1,epochs+1), val_accuracy, color='red', lw=3, label='validation')
plt.xlabel('epochs')
plt.ylabel('rmse')
plt.show()

train_accuracy = history.history['r2_metric']
val_accuracy = history.history['val_r2_metric']
plt.plot(range(2,epochs+1), train_accuracy[1:], color='blue', lw=3, label='training')
plt.plot(range(1,epochs+1), val_accuracy, color='red', lw=3, label='validation')
plt.xlabel('epochs')
plt.ylabel('r2')
plt.show()
'''
