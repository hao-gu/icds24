# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:34:13 2023

@author: haofg
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#lon
xmin = -74
xmax = -35

#lat
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
data[data<=0] = np.nan

#data distr check
plt.hist(data,bins=100)
plt.show()

np.round(lat, 1, lat)
np.round(lon, 1, lon)

#visualize
plt.scatter(lon, lat, s = 1, c = data, cmap = "coolwarm")
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()

points = dict()#index dictionary
for i in range(len(lon)): #adding points to dictionary
    index = (lon[i], lat[i]) #index is like the key i assign each grid box
    if index in points:
        points[index].append(data[i])
    else:
        points[index] = [data[i]]

#restrcutured data
data_new = []
lon_new = []
lat_new = []
for key in points: #key is the coordinate tuple
    lon_new.append(key[0])
    lat_new.append(key[1])
    temp = np.array(points[key])
    data_new.append(np.nanmean(temp)) #get average
data_new = np.array(data_new)
lon_new = np.array(lon_new)
lat_new = np.array(lat_new)
#visualize
plt.scatter(lon_new, lat_new, s = 1, c = data_new, cmap = "coolwarm")
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()
#histogram
plt.hist(data_new,bins=100)
plt.show()