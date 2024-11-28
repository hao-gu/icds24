# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:41:13 2023

@author: haofg
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

with open('allfiles.txt', 'r') as file:
    i=0
    for line in file:
        i+=1
        print(i)
        filename = "C:\\Users\\haofg\\Documents\\Remote Sensing Code\\Amazon GEDI Data\\" + line[87:-1] 
        #filename = "C:/Users/haofg/Documents/Remote Sensing Codes/Amazon GEDI Data/" + line[87:-1]
        #print(filename)
        f = h5py.File(filename,'r')
        '''
        #keys = ['BEAM0000'] #,'BEAM0001','BEAM0010','BEAM0011','BEAM0101','BEAM0110','BEAM1000','BEAM1011']
        #for key in keys:
        key = "BEAM0000"
        group = f[key]
        gedi = group['agbd'][:]
        lat = group['lat_lowestmode'][:] 
        lon = group['lon_lowestmode'][:] #
        time = group['delta_time'][:]
        #gedi[gedi<0] = np.nan
        plt.scatter(lon,lat,s=1, c=gedi, cmap='coolwarm')
        plt.ylim(-33, 5)
        plt.xlim(-74, -35)
        '''
        f.close()
    plt.colorbar()
    plt.show()

#"C:\Users\haofg\Documents\Remote Sensing Codes\codes\Amazon GEDI Data"
#C:\Users\haofg\Documents\Remote Sensing Code\Amazon GEDI Data
#"C:\Users\haofg\Documents\Remote Sensing Code\Amazon GEDI Data"