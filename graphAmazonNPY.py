# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:38:25 2024

@author: haofg

To plot saved npy GEDI and ERA5 files
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd

countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color = 'lightgrey')

###### constants ########
#lon
xmin = -80
xmax = -43

#lat
ymin = -20
ymax = 10
#########################
plt.style.use('default')

lon = np.load('gedi_lon1.npy')
lat = np.load('gedi_lat.npy')
data = np.load('gedi_agbd.npy')
#visualize


#########limits#########
plt.scatter(lon, lat, s = 2, c = data, cmap = "viridis",vmin=0, vmax=200) # max is 2800, limiting for more color contrast
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()

####### whole world w limits ###########
africa = countries[countries["continent"] == "South America"]
africa.plot(color = 'lightgrey')
plt.scatter(lon, lat, s = 0.5, c = data, cmap = "viridis",vmin=0, vmax=200) # max is 2800, limiting for more color contrast
plt.gca().add_patch(Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
plt.colorbar()
plt.ylim(-60, 30)
plt.xlim(-90, -25)
plt.show()