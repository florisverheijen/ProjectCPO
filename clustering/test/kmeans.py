# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:02:26 2020

@author: s152040
"""
#import all libraries you need here:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#initialisation 
df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 72, 70, 79, 68, 65],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 42, 45, 62, 55, 59]
})  # simple dataset to start of with later this should be replaced for the comumns we have in our big dataset

# this makes sure that the random numbers are the same every time you run it can be deleted when the dataset is found
np.random.seed(200) 

k = 3 #number of centroids
# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
    for i in range(k)
}
  
 #make a figure to make it visible  
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b',4:'m'} # so you can see the centroids you just created and can assign this color to the other points
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# assining the points to a cluster

#build a function that does this so you can put in your centroids and data. 
#and it works for multiple datasets

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids) #check if it worked by printing the first 5/head of the dataFrame
print(df.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5) # alpha is set to 0.5 to show the difference between a normal point and the cluster point 
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# now you need to update this so that the cluster points go to a better suited location.
#after this you can also optimise for better results 


def update(k): #here you update the centroid location to the mean of all points that were in that cluster so you differentiate even more
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)

# now this is done we can repeat the assignment of the datapoints to make the clusters better. 
# because we made a function of the assignment we dob't have to do it all again.
df = assignment(df,centroids)
#here the points are assigned again a blue point is now red as can be seen if you plot it 



# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

#you can see that the clusters change a bit now we want to do this until nothing changes anymore and it is optimised.

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centrouds = update(centroids)
    df=assignment(df,centroids)
    if closest_centroids.equals( df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

# this loop makes sure that everything becomes stable and optimises for all the training data you have.
# the full programm should have this loop but without the steps in between but they are for educational purposes.
"""
full program finished :
"""
















