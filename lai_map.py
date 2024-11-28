    # -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:05:33 2023

@author: jgwir
"""
#saves ERA5 data on amazon into NPZ files

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import netCDF4 as nc
#import h5py 
import math
# t2m, ssrd, tp, sp
 #read in hdf5
#with open('nctimeavg.npz', 'rb') as file:
    #npzfile = np.load(file) #load data
    #npzfile.files
    #npzfile['lai_hv']
#x = np.linspace(1,10,10)
#y = np.linspace(10,99,30)
#np.savez('test.npz', x=x, y=y)
'''
data = np.load('nctimeavg.npz')
mask = np.load('nctimeavg_ma.npz')
for varname, varmask in zip(data.files, mask.files):
    temp = ma.array(data[varname], mask = mask[varmask])
    #temp[temp < 0] = np.nan
    plt.imshow((temp))#[850:1230,2860:3250]) #10N80W, 20S40W 
    plt.colorbar()
    plt.title(varname)
    plt.show()
'''
#congo
xmin = 3
xmax = 30

ymin = -7
ymax = 7


#[800 - int(Y[j] * 10)][(3600 + int(X[i] * 10)) % 3600]
#lai
files = ["lai",'temperature', 'rain', 'radiation','pressure','wind']
varnames = ['lai_hv','t2m','tp','ssrd','sp','u10']
'''
results = []
for file, varname in zip(files, varnames):
    ds = nc.Dataset('C:\\Users\\haofg\\Documents\\Remote Sensing Code\\codes\\data\\' + file + '.nc')
    result = ds[varname][:, 900 - ymax * 10 : 900 - ymin * 10, (3600 + xmin * 10) % 3600 : (3600 + xmax * 10) % 3600]
    result = np.nanmean(result, axis = 0)
    plt.imshow(result) #10N80W, 20S40W 
    plt.colorbar()
    plt.title(varname)
    plt.show()  
    results.append(result) 
with open('nctimeavg.npz', 'wb+') as file:
    np.savez(file, lai_hv = results[0], temperature = results[1],
             rain = results[2], radiation = results[3],
             pressure = results[4], wind = results[5])
with open('nctimeavg_ma.npz', 'wb+') as file:
    np.savez(file, lai_hvma = results[0].mask, 
             t_ma = results[1].mask,
             rain_ma = results[2].mask,
             radiation_ma = results[3].mask,
             pressure_ma = results[4].mask,
             wind_ma = results[5].mask)
  ''' 

'''
#lai
ds = nc.Dataset('../../data/lai.nc')
varname = 'lai_hv'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])

for i in range(1,240):
    temp = ds[varname][i]
    temp[temp<0] = 0
    result += temp #result * (result * i + ds[varname][i]) / ((i + 1) * result);
    print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.title(varname)
plt.show()
temperature = result/20

#temperature 
ds = nc.Dataset('data/temperature.nc')
varname = 't2m'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])
for i in range(1,240):
    if i%12 == 7:
        temp = ds[varname][i]
        temp[temp<0] = 0
        result += temp# result * (result * i + ds[varname][i]) / ((i + 1) * result);
        print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.title(varname)
plt.show()
temperature = result/20

#rain
ds = nc.Dataset('data/rain.nc')
varname = 'tp'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])
for i in range(1,240):
    if i%12 == 7:
        temp = ds[varname][i]
        temp[temp<0] = 0
        result += temp# result * (result * i + ds[varname][i]) / ((i + 1) * result);
        print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.title(varname)
plt.show()
rain = result/20

#radiation
ds = nc.Dataset('data/radiation.nc')
varname = 'ssrd'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])
for i in range(1,240):
    if i%12 == 7:
        temp = ds[varname][i]
        temp[temp<0] = 0
        result += temp# result * (result * i + ds[varname][i]) / ((i + 1) * result);
        print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.title(varname)
plt.show()
radiation = result/20

#pressure
ds = nc.Dataset('data/pressure.nc')
varname = 'sp'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])
for i in range(1,240):
    if i%12 == 7:
        temp = ds[varname][i]
        temp[temp<0] = 0
        result += temp# result * (result * i + ds[varname][i]) / ((i + 1) * result);
        print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.clim(0,1e5)
plt.title('surface pressure')
plt.show()
pressure = result/20

#wind
ds = nc.Dataset('data/wind.nc')
varname = 'u10'
result = ds[varname][0] # np.array([[1,2,3],[4,5,6]])
for i in range(1,240):
    if i%12 == 7:
        temp = ds[varname][i]
        temp[temp<0] = 0
        result += temp# result * (result * i + ds[varname][i]) / ((i + 1) * result);
        print(i)
plt.imshow((result/20)[800:1100,2800:3200]) #10N80W, 20S40W 
plt.colorbar()
plt.title(varname)
plt.show()
wind = result/20

with open('nctimeavg.npz', 'wb') as file:
    np.savez(file, lai_hv = lai_hv, temperature = temperature,
             rain = rain, radiation = radiation,
             pressure = pressure, wind = wind)
with open('nctimeavg_ma.npz', 'wb') as file:
    np.savez(file, lai_hvma = lai_hv.mask, 
             t_ma = temperature.mask,
             rain_ma = rain.mask,
             radiation_ma = radiation.mask,
             pressure_ma = pressure.mask,
             wind_ma = wind.mask)
'''

'''
x = np.linspace(0,180,181)
y = np.linspace(0,359,360)

grid_new = []
for i in range(len(x)):
    for j in range(len(y)):
        grid_new.append([x[i], y[j]])
grid_new = np.array(grid_new)

lon = ds['longitude'][:]
lat = ds['latitude'][:]
grid = [] #with lon/lat coord points
for i in range(len(lon)):
    for j in range(len(lat)):
        grid.append([lon[i], lat[j]])
grid = np.array(grid)

data = ds['sp'][0].flatten('C') #check example code

data_new = griddata(grid, data, grid_new, method = 'cubic')

plt.imshow(data_new.reshape(181,360))
plt.title('interpolated surface presssure')
plt.colorbar()
plt.show()

#np.savez(varname)
'''