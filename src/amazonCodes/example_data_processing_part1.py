import numpy as np
import matplotlib.pyplot as plt
import h5py    

# research task
# 1) compare different beam biomass density estimate (visualize)

# More GEDI data
import requests
import datetime as dt 
import pandas as pd
import geopandas as gpd
import contextily as ctx 
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import orient

doi = '10.3334/ORNLDAAC/2056'# GEDI L4A DOI 

# CMR API base url
cmrurl='https://cmr.earthdata.nasa.gov/search/' 

doisearch = cmrurl + 'collections.json?doi=' + doi
response = requests.get(doisearch)
response.raise_for_status()
concept_id = response.json()['feed']['entry'][0]['id']

print(concept_id)

# Brazil bounding box
bound = (-74, -33, -35, 5) 
#Congo bounding box
#bound = (3, -7, 30, 7) 

# time bound
start_date = dt.datetime(2020, 7, 1) # specify your own start date
end_date = dt.datetime(2020, 7, 31)  # specify your end start date

# CMR formatted start and end times
dt_format = '%Y-%m-%dT%H:%M:%SZ'
temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)

# CMR formatted bounding box
bound_str = ','.join(map(str, bound))

page_num = 1
page_size = 2000 # CMR page size limit

granule_arr = []

while True:
    
    # defining parameters
    cmr_param = {
        "collection_concept_id": concept_id, 
        "page_size": page_size,
        "page_num": page_num,
        "temporal": temporal_str,
        "bounding_box[]": bound_str
    }
    
    granulesearch = cmrurl + 'granules.json'

    response = requests.get(granulesearch, params=cmr_param)
    response.raise_for_status()
    granules = response.json()['feed']['entry']
    
    if granules:
        for g in granules:
            granule_url = ''
            granule_poly = ''
            
            # read file size
            granule_size = float(g['granule_size'])
            
            # reading bounding geometries
            if 'polygons' in g:
                polygons= g['polygons']
                multipolygons = []
                for poly in polygons:
                    i=iter(poly[0].split (" "))
                    ltln = list(map(" ".join,zip(i,i)))
                    multipolygons.append(Polygon([[float(p.split(" ")[1]), float(p.split(" ")[0])] for p in ltln]))
                granule_poly = MultiPolygon(multipolygons)
            
            # Get URL to HDF5 files
            for links in g['links']:
                if 'title' in links and links['title'].startswith('Download') \
                and links['title'].endswith('.h5'):
                    granule_url = links['href']
            granule_arr.append([granule_url, granule_size, granule_poly])
               
        page_num += 1
    else:
        break

# adding bound as the last row into the dataframe
# we will use this later in the plot
b = list(bound)
granule_arr.append(['bound', 0, box(b[0], b[1], b[2], b[3])]) 

# creating a pandas dataframe
l4adf = pd.DataFrame(granule_arr, columns=["granule_url", "granule_size", "granule_poly"])

# Drop granules with empty geometry
l4adf = l4adf[l4adf['granule_poly'] != '']

print ("Total granules found: ", len(l4adf.index)-1)
print ("Total file size (MB): ", l4adf['granule_size'].sum())

# creating a geopandas dataframe
gdf = gpd.GeoDataFrame(l4adf, geometry=l4adf.granule_poly)
# assiging CRS as WGS84
gdf.crs = "EPSG:4326"
# converting CRS to Web Mercator, the reference system of the contextily basemap
gdf_epsg3857 = gdf.to_crs(epsg=3857)
# plotting GEDI suborbits
ax=gdf_epsg3857[:-1].plot(figsize=(10, 5), alpha=0.3, edgecolor='green')
# plotting the bounding box (last row)
gdf_epsg3857[-1:].plot(ax=ax, facecolor='none', edgecolor='red')
# adding to contextily basemap
ctx.add_basemap(ax)

# part II multiple files
# drop duplicate URLs if any
l4a_allfiles = l4adf[:-1].drop_duplicates(subset=['granule_url'])
l4a_allfiles.to_csv('congofiles.txt', columns = ['granule_url'], index=False, header = False)
