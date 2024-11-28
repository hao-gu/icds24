# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:07:07 2024

@author: haofg

congo DATA
Interpolates ERA5 and GEDI data to match
combines GEDI aggregate alg and turns ERA5 into a list 
"""

import numpy as np
import numpy.ma as ma
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

xmin = 3
xmax = 30

ymin = -7
ymax = 7

#UNCOMMENT
#gedi long lat 
grid_new = []
lon = []
lat = []
data = []

print('started reading gedi lon/lat')
with open('congofiles.txt', 'r') as file:
    i=0
    for line in file:
        i+=1
        print(i)
        filename ='C:\\Users\\haofg\\Documents\\Remote Sensing Code\\congoGEDI\\' + line[87:-1]
        f = h5py.File(filename, 'r')
        keys = ['BEAM0000','BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110','BEAM1000','BEAM1011']
        for key in keys:
            try:
                beam = f[key]
                lat = np.append(lat, beam['lat_lowestmode'][:]) 
                lon = np.append(lon, beam['lon_lowestmode'][:])
                data = np.append(data, beam['agbd'][:])
            except:
                print(key + " doesnt exist")
################### agregate data points within a file
data[data<=0] = np.nan
#check distr
plt.hist(data,bins=100)
plt.show()

lat = np.round(lat*4, 0)/4
lon = np.round(lon*4, 0)/4

points = dict()#index dictionary
for i in range(len(lon)): #adding points to dictionary
    index = (lon[i], lat[i]) #index is like the key i assign each grid box
    #uncomment later
    #if index[0] > xmax or index[0] < xmin or index[1] > ymax or index[1] < ymin: # ONLY INCLUDE IN BOUNDS
     #   continue
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
    data_new.append(np.nanmean(np.array(points[key]))) #get average
data_new = np.array(data_new)
lon_new = np.array(lon_new)
lat_new = np.array(lat_new)

#check data distr
plt.hist(data_new,bins=100)
plt.show()

'''
############################ drop nan???? DONT USE
inputs = ['gedi_lon_025','gedi_lat_025','gedi_agbd_025']
X = np.empty([199526])
for i in inputs:
    f = np.load(i + '.npy', 'r')
    X = np.vstack((X, f))
X = pd.DataFrame(X[1:, :].T, columns=inputs)
X = X.dropna()
'''

'''
########################## get rid of nan in lat/lon :( 

lon_copy = lon_new.copy()
lat_copy = lat_new.copy()
data_copy = data_new.copy()

lon_new=[]
lat_new=[]
data_new=[]
for i in range(len(lon_copy)):
    if not np.isnan(lon_copy[i]) and not np.isnan(lat_copy[i]):
        lon_new.append(lon_copy[i])
        lat_new.append(lat_copy[i])
        data_new.append(data_copy[i])
data_new = np.array(data_new)
lon_new = np.array(lon_new)
lat_new = np.array(lat_new)

####################grid new for interp purposes
grid_new = np.vstack((lon_new, lat_new)).transpose() #???????
#grid_new = np.load('grid_new.npy') #???????

'''
#####################################save gedi data npys 
with open('gedi_lon_congo_display.npy', 'wb') as f: #used lat1 bc lat was fricked up
    np.save(f, lon_new) 
 #xomethings wrong with longitude file
with open('gedi_lat_congo_display.npy', 'wb') as f:
    np.save(f, lat_new) 
with open('gedi_agbd_congo_display.npy', 'wb') as f:
    np.save(f, data_new) 
with open('grid_new_025.display', 'wb') as f:
    np.save(f, grid_new) 
print('done')



'''
############# visualize
plt.scatter(lon,lat,s=0.2, c=data, cmap='coolwarm')
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()
'''
'''
lon_copy = lon_new.copy()
lat_copy = lat_new.copy()
data_copy  = data_new.copy()
'''

'''
############Load GEDI data##############
lon_new = np.load('gedi_lon_congo.npy') #used lat1 bc lat was fricked up
lat_new = np.load('gedi_lat_congo.npy')
data_new = np.load('gedi_agbd_congo.npy')
grid_new = np.load('grid_new.npy')
print('done loading')
##########################



######################ERA5 Grid
X = np.linspace(xmin, xmax, 271) # replace with ERA5 longitude
print(X)
Y = np.linspace(ymin, ymax, 141) # replace with ERA5 latitude
print(Y)
#get july data  
#currently using avg data
data_nc = np.load('nctimeavg.npz')
mask = np.load('nctimeavg_ma.npz')
for varname, varmask in zip(data_nc.files, mask.files):
    print('started reading nc data ' + varname)
    temp = ma.array(data_nc[varname], mask = mask[varmask]) # creates new masked np array with npy files
    temp = ma.filled(temp, fill_value = np.nan) #fill with all nans
    grid = []
    values = []
    for i in range(len(X)):
        for j in range(len(Y)):
            grid.append([X[i], Y[j]])
            values.append(temp[900 - int(Y[j] * 10)][(3600 + int(X[i] * 10)) % 3600]) 
    grid = np.array(grid)
    print('done reading nc data')
    print('start interpolating '+ varname)
    era5_data_new = griddata(grid, values, grid_new, method='nearest') 
    np.save(varname + '_interp_congo.npy', era5_data_new)    
    print('done interp')
    plt.scatter(grid_new[:,0], grid_new[:,1], s=0.5, c=era5_data_new, cmap='viridis')
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.colorbar()
    plt.show()
'''