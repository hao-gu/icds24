# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:38:25 2024

@author: haofg

To plot saved npy GEDI for Congpo and amazon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd

countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color = 'lightgrey')

###### constants ########
xmin = 3
xmax = 30

ymin = -7
ymax = 7
#########################

plt.style.use('default')

lon = np.load('gedi_lon_congo_display.npy')
lat = np.load('gedi_lat_congo_display.npy')
data = np.load('gedi_agbd_congo_display.npy')
#visualize


#########limits#########
plt.scatter(lon, lat, s = 2, c = data, cmap = "viridis",vmin=0, vmax=200) # max is 2800, limiting for more color contrast
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.colorbar()
plt.show()

####### whole world w limits ###########
africa = countries[countries["continent"] == "Africa"]
countries.plot(color = 'lightgrey')
plt.scatter(lon, lat, s = 0.5, c = data, cmap = "viridis",vmin=0, vmax=200) # max is 2800, limiting for more color contrast
plt.gca().add_patch(Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
plt.colorbar()#.set_ylabel('biomass Mg/ha', rotation=270)
plt.xlim(-25, 55)
plt.ylim(-40, 40)
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.show()