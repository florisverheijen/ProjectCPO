# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:50:14 2020

@author: s152040
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  silhouette_score


#initialisation 
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 72, 70, 79, 68, 65],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 42, 45, 62, 55, 59]
})  # simple dataset to start of with later this should be replaced for the comumns we have in our big dataset

# this makes sure that the random numbers are the same every time you run it can be deleted when the dataset is found
np.random.seed(200) 

clusterrange = [2, 3, 4, 5, 6]
silhouette_list=[]
for k in clusterrange:
    
    
# centroids[i] = [x, y]
    centroids = {
        i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
        for i in range(k)
    }
    
    colmap = {1: 'r', 2: 'g', 3: 'b',4: 'm', 5: 'c',6: 'y'} # so you can see the centroids you just created and can assign this color to the other points
    
    def assignment(df, centroids):
        for i in centroids.keys():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2) euclidian distance 
            df['distance_from_{}'.format(i)] = (
                np.sqrt(
                    (df['x'] - centroids[i][0]) ** 2 + (df['y'] - centroids[i][1]) ** 2
                )
            )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        df['color'] = df['closest'].map(lambda x: colmap[x])
        return df
    
    df = assignment(df, centroids) #assign it the first time
    
    def update(k): #here you update the centroid location to the mean of all points that were in that cluster so you differentiate even more
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        return k
    
    
    while True:
        closest_centroids = df['closest'].copy(deep=True)
        centrouds = update(centroids)
        df=assignment(df,centroids)
        if closest_centroids.equals( df['closest']):
            break
    
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5)
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()    
    coords = []
    clust = []
    
    for t in range(len(df)):
        loccoord = []
        loccoord.append(df['x'].iloc[t])
        loccoord.append(df['y'].iloc[t])
        clust.append(df['closest'].iloc[t])
        coords.append(loccoord)
    
    
    silhouette_avg = silhouette_score(coords, clust)
    print(silhouette_avg)
    silhouette_list.append(silhouette_avg)


fig = plt.figure(figsize=(5, 5))
plt.plot(clusterrange,silhouette_list,'o-')
plt.xlim(1.5, 6.5)
plt.ylim(0, 1)
plt.show()


