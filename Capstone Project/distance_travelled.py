#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from geopy.distance import vincenty
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='pulks', api_key='Wf9whq6hQkWW4Va2OgEc')
import pylab

# load the dataset

line = plt.figure()
dataset = pandas.read_csv('pops.csv')
dataset = dataset[['individual-local-identifier','location-long','location-lat']] 
dataset = np.asarray(dataset)
dist_travelled = []
dist_bird = 0

#calculate the distance travelled by each bird
for i in range(len(dataset)-1):
    
    if dataset[i,0] == dataset[i+1,0]:
        dist_bird = dist_bird + (vincenty((dataset[i,1],dataset[i,2]), (dataset[i+1,1],dataset[i+1,2])).miles)
    else:
        dist_travelled.append(dist_bird)
        dist_bird = 0
x = np.zeros((len(dist_travelled)))
for i in range(len(x)):
    x[i] = i+1
print x
print len(x)
dist_travelled = np.asarray(dist_travelled)
print dist_travelled.shape
plt.title('Distance travelled by each bird during their migratory journey')
plt.xlabel('Bird Number')
plt.ylabel('Distance travelled in km')
plt.bar(x,dist_travelled,alpha=0.4,yerr=np.std(dist_travelled),
                 color='r',)
pylab.savefig('distance_travlled.png',dpi=1000)
plt.show()

plot_url = py.plot_mpl(line, filename='mpl-docs/add-line')