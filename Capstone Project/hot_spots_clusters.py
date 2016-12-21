#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.dates as md
import dateutil
import pylab
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='pulks', api_key='Wf9whq6hQkWW4Va2OgEc')


df = pd.read_csv('vegetation_cover.csv')
print df.head()



fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
 
m = Basemap(projection='merc', llcrnrlat=-30., urcrnrlat=70.,
            llcrnrlon=-30., urcrnrlon=90., resolution='i')

m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m.drawmapboundary(fill_color='#000000')                # black background
m.drawcountries(linewidth=0.1, color="w")
  
x, y = m(df['location-long'].values, df['location-lat'].values)

'''
Plot data on the map according to the marker size
of either population density or vegetation cover
'''


#plt_data = np.genfromtxt("polpulation_new.csv", delimiter=",")
#df['MODIS Ocean Aqua OceanColor 4km Daily PIC'] = df['MODIS Ocean Aqua OceanColor 4km 8d PIC'].fillna(method='ffill')
#s=df['SEDAC GRUMP v1 2000 Population Density Adjusted']*10
m.scatter(x, y, c="#1292db", lw=0, alpha=1, zorder=5, s=df['ECMWF Interim Full Daily Invariant High Vegetation Cover']*10)

plt.title("Vegetation Cover of birds during their migratory journey")
pylab.savefig('veg.png',dpi=1000)

plt.show()

