# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:34:13 2023

@author: haofg

FOR AMAZON DATA
Interpolates ERA5 and GEDI data to match
combines GEDI aggregate alg and turns ERA5 into a list 
"""
import numpy as np
import numpy.ma as ma
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

'''
old coordinate box
xmin = -74
xmax = -35

ymin = -33
ymax = 5
'''
###### constants ########
#lon
xmin = -80
xmax = -43

#lat
ymin = -20
ymax = 10
'''
#gedi long lat 
grid_new = []
lon = []
lat = []
data = []

print('started reading gedi lon/lat')
    #allfiles in codes Specific Data folder
with open('../../AmazonGediData3/allfiles.txt', 'r') as file:
    i=0
    for line in file:
        i+=1
        print(i)
        filename ='C:\\Users\\haofg\\Documents\\Remote Sensing Code\\AmazonGediData3\\' + line[87:-1]
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

#aggreagte to 0.25 resolution
lat = np.round(lat*4, 0)/4
lon = np.round(lon*4, 0)/4

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

#check data distr
plt.hist(data_new,bins=100)
plt.show()

grid_new = np.vstack((lon_new, lat_new)).transpose()
#grid_new = np.array(grid_new) #???????
'''
#remove nan
'''
lon_copy = lon_new.copy()
lat_copy = lat_new.copy()
data_copy = data_new.copy()
'''
'''
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
grid_new = np.vstack((lon_new, lat_new)).transpose()
'''
'''
#####################################save gedi data npys 
# need to convert back to numpy arrays, and fukcing save it thanks and update grid_new AGHFHFHHFHFHF
with open('gedi_lon_025.npy', 'wb') as f:
    np.save(f, lon_new) 
 #xomethings wrong with longitude file
with open('gedi_lat_025.npy', 'wb') as f:
    np.save(f, lat_new) 
with open('gedi_agbd_025.npy', 'wb') as f:
    np.save(f, data_new) 
with open('grid_new_amazon_025.npy', 'wb') as f:
    np.save(f, grid_new) 
print('done')

'''

#load data
grid_new = np.load('grid_new_amazon_025.npy')
#lon_new = np.load(('gedi_lon_025.npy'))
#lat_new = np.load('gedi_lat_025.npy')
#data_new = np.load('gedi_agbd_025.npy')

#plt.scatter(lon_new,lat_new, s=1, c=data_new)
#plt.show()

######################ERA5 Grid
X = np.arange(xmin, xmax+0.1, 0.1) # replace with ERA5 longitude
#print(X)
Y = np.arange(ymin, ymax+0.1, 0.1) # replace with ERA5 latitude
#print(Y) 

#get era5 data

data_nc = np.load('nctimeavg.npz')
mask = np.load('nctimeavg_ma.npz')
for varname, varmask in zip(data_nc.files, mask.files):
    print('started reading nc data ' + varname)
    #reconstrtuct masked array
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
    data_new = griddata(grid, values, grid_new, method='nearest') 
    with open(varname + '_interp_025.npy', 'wb') as f:
        np.save(f, data_new)    
    print('done interp')
    plt.scatter(grid_new[:,0], grid_new[:,1], s=0.2, c=data_new, cmap='coolwarm')
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.colorbar()
    plt.show()
