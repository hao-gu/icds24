# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

filename = "GEDI04_A_2019108015253_O01960_02_T03910_02_002_02_V002.h5"
f = h5py.File(filename,'r')

key="BEAM0001"
group = f[key]
for var in group.keys():
    print(var)

gedi = group['agbd'][:]
lat = group['lat_lowestmode'][:] #why are they turned into numpy arrays...
lon = group['lon_lowestmode'][:] #
time = group['delta_time'][:]

mean_lon = np.array([])
mean_lat = np.array([])
mean_gedi = np.array([])
index = 0
current_sum = 0
current_lon = round(lon[0],1)
current_lat = round(lat[0],1)

gedi[gedi<0] = np.nan

for i in range(0, gedi.size):
    if (round(lon[i], 1) != current_lon or round(lat[i], 1) != current_lat or gedi.size - 1 == i):
        if (index != 0):
            print('lol')
            mean_lon = np.append(mean_lon, current_lon)
            mean_lat = np.append(mean_lat, current_lat)
            mean_gedi = np.append(mean_gedi, float(current_sum/index))
        current_lon = round(lon[i], 1)
        current_lat = round(lat[i], 1)
        current_sum = 0
        index = 0
    if math.isnan(gedi[i]) == False:
        index += 1 
        current_sum += gedi[i]
        print(gedi[i], index)
#plt.hist(gedi,bins=100)
#for (int i=0)

plt.scatter(mean_lon,mean_lat,s=mean_gedi, c=mean_gedi, cmap='coolwarm')
plt.colorbar()
plt.show()
plt.scatter(lon,lat,s=gedi, c=gedi, cmap='coolwarm')
plt.colorbar()
plt.show()

f.close()
