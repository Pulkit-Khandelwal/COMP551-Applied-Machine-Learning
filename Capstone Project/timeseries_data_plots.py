#!/usr/bin/env python2
# -*- coding: utf-8 -*-

Created on Sat Dec  3 17:03:07 2016

@author: pulkit


import pandas
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='pulks', api_key='Wf9whq6hQkWW4Va2OgEc')
dataset = pandas.read_csv('gps.csv', engine='python')
print dataset.columns
df = pandas.DataFrame(dataset, columns = ['timestamp', 'location-long'])
print df.shape
plt.plot(df)
plt.show()




import matplotlib.dates as md
import dateutil
import numpy as np
import pandas
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='pulks', api_key='Wf9whq6hQkWW4Va2OgEc')


dataset = pandas.read_csv('pic.csv', engine='python')

dataset['timestamp'] = pandas.to_datetime(dataset.timestamp)
dataset = dataset.sort('timestamp')

df = pandas.DataFrame(dataset, columns = ['timestamp'])
datestrings = dataset['timestamp'].tolist()
print datestrings[0]

dataset[['SEDAC GRUMP v1 2000 Population Density Adjusted']] = dataset[['SEDAC GRUMP v1 2000 Population Density Adjusted']].apply(pandas.to_numeric)


plt_data = dataset['SEDAC GRUMP v1 2000 Population Density Adjusted'].tolist()

for i in range(len(plt_data)):
    if plt_data[i]==None:
        plt_data[i] = plt_data[i-1]
    
print plt_data[0]

ax=plt.gca()

#plotting the time series data: 1

xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
plt.plot(dates,plt_data)
plt.show()

data = [go.Scatter(x=datestrings,y=plt_data)]
py.iplot(data)



df = pd.read_csv('pic.csv')
print df.head()

#plotting the time series data: 2


scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'world',
        lon = df['location-long'],
        lat = df['location-lat'],
        text = df['SEDAC GRUMP v1 2000 Population Density Adjusted'],
        mode = 'markers',
        marker = dict( 
            size = 8, 
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['SEDAC GRUMP v1 2000 Population Density Adjusted'],
            cmax = df['SEDAC GRUMP v1 2000 Population Density Adjusted'].max(),
            colorbar=dict(
                title="Incoming flightsFebruary 2011"
            )
        ))]

layout = dict(
        title = 'Most trafficked US airports<br>(Hover for airport names)',
        colorbar = True,   
        geo = dict(
            scope='world',
            projection=dict( type='world' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5        
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-airports' )


plt_data = np.genfromtxt("pic.csv", delimiter=",",dtype=np.float64)
plt_data = plt_data[1:,14]
print np.nanmax(plt_data)

df = pd.read_csv('pic.csv')
df.head()

df['text'] =  '<br>Population Density ' + (df['SEDAC GRUMP v1 2000 Population Density Adjusted']).astype(str)
limits = [(0,100),(100,1000),(1000,30000)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)"]
cities = []
scale = 500

#plotting the time series data: 3


for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    city = dict(
        type = 'scattergeo',
        locationmode = 'europe| africa',
        lon = df_sub['location-long'],
        lat = df_sub['location-lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['SEDAC GRUMP v1 2000 Population Density Adjusted']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]) )
    cities.append(city)

layout = dict(
        title = 'Population Density',
        showlegend = True,
        geo = dict(
            scope='europe| africa',
            projection=dict( type='europe| africa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-populations' )
