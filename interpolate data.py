# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:34:13 2023
FAILURE DONT USE 
@author: haofg
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

xmin = -74
xmax = -35

ymin = -33
ymax = 5

#example gedi data 
filename ='C:\\Users\\haofg\\Documents\\Remote Sensing Code\\Amazon GEDI Data\\GEDI04_A_2020183013239_O08785_01_T00012_02_002_02_V002.h5'
f = h5py.File(filename, 'r')
key = 'BEAM0001'
beam = f[key]
data = beam['agbd'][:]
data[data<0] = np.nan
lat = beam['lat_lowestmode'][:] 
lon = beam['lon_lowestmode'][:]
np.round(lat, 1)
np.round(lon, 1)
print('start')
grid = []
for i in range(len(lon)):
    grid.append([lon[i],lat[i]])
        #data append
grid = np.array(grid)
print('donr')
plt.scatter(lon,lat,s=0.2, c=data, cmap='coolwarm')
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()
print('start2')
#ERA5 Grid
X = np.linspace(xmin, xmax, 391) # replace with ERA5 longitude
print(X)
Y = np.linspace(ymin, ymax, 381) # replace with ERA5 latitude
print(Y)

grid_new = []
for i in range(len(X)):
    for j in range(len(Y)):
        grid_new.append([X[i], Y[j]])
grid_new = np.array(grid_new)
print('done2')
data_linear = griddata(grid, data, grid_new, method='linear') # X: ERA5 temperature at GEDI gridbox
data_cubic = griddata(grid, data, grid_new, method='cubic')   
data_nn = griddata(grid, data, grid_new, method='nearest')
print('done3')
#data_new = np.load('data_new.npy')
data_linear.resize(391,381)
plt.imshow(data_linear)    
plt.colorbar()
plt.title("interpolated")
plt.show()
data_cubic.resize(391,381)
plt.imshow(data_cubic)    
plt.colorbar()
plt.title("interpolated")
plt.show()
data_nn.resize(391,381)
plt.imshow(data_nn)    
plt.colorbar()
plt.title("interpolated")
plt.show()
print('done4')
