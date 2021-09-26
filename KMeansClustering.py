# Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import joblib

# Import dataset
data = pd.read_csv('TestData.csv')

# Set dates to datetime objects
data['starttime'] = pd.to_datetime(data['starttime'], dayfirst=True)
data['stoptime'] = pd.to_datetime(data['stoptime'], dayfirst=True)

# Add weekday column to dataset
data['weekday'] = data['starttime'].dt.dayofweek
data['starttime'] = data['starttime'].dt.round('1H')
# Seperate data to smaller dataset only including weekday and starttime
data = data[['starttime', 'weekday', 'start station latitude', 'start station longitude']]
# Count demand in 1 hour timeframe by grouping on same rows. Counted demand per hour on weekday
data = data.groupby(['starttime', 'weekday', 'start station latitude', 'start station longitude']).size().reset_index(name='count')

# it is chosen to ontly take hour and weekday into account as we only have data for 1 month of a single year
data['start_hour'] = data['starttime'].dt.hour
data.drop('starttime', axis=1, inplace=True)
first_column = data.pop('count')
data.insert(0, 'count', first_column)

#%%
# Set data to array
X = data.iloc[:, :].values

# Elbow method to find optimal number of clusters
elbow = []
for i in range(1, 11):
    kMeans = KMeans(n_clusters=i, init='k-means++')
    kMeans.fit(X)
    elbow.append(kMeans.inertia_)
plt.plot(range(1, 11), elbow)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel("WCSS")
plt.show()

#%%
# Training the K-Means model on the dataset try # of cluster 3 and 5
db_list = []
ic_list = []
sil_list = []

finalkmeans = KMeans(n_clusters=5, max_iter=10000)  # n = based on elbow
finalkmeans.fit(X)
finalkmeans.inertia_
finalkmeans.cluster_centers_
finalkmeans.n_iter_
finalkmeans.labels_[:4]
y_pred = finalkmeans.predict(X)
pred = pd.DataFrame(y_pred)
pred.columns = ['ClusterType'] #store K-means results in a dataframe
X = pd.DataFrame(X, columns=['count', 'weekday', 'start station latitude', 'start station longitude', 'start_hour'])
prediction = pd.concat([X, pred], axis=1)

#%%
# Making the grouping visual based on weekday, starthour, and count
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(prediction["start_hour"], prediction["weekday"], prediction["count"],
           linewidths=1, alpha=.7,
           edgecolor='k',
           s=33,
           c=prediction["ClusterType"])
plt.xlabel('start hour')
plt.ylabel('weekday')
ax.set_zlabel('count')
ax.view_init(5, 85)
ax.legend()
plt.show()

# Visualization of the clusters based on location
plt.scatter(prediction.loc[prediction['ClusterType'] == 0, 'start station latitude'], prediction.loc[prediction['ClusterType'] == 0, 'start station longitude'], s=33, c='red')
plt.scatter(prediction.loc[prediction['ClusterType'] == 1, 'start station latitude'], prediction.loc[prediction['ClusterType'] == 1, 'start station longitude'], s=33, c='green')
plt.scatter(prediction.loc[prediction['ClusterType'] == 2, 'start station latitude'], prediction.loc[prediction['ClusterType'] == 2, 'start station longitude'], s=33, c='yellow')
plt.scatter(prediction.loc[prediction['ClusterType'] == 3, 'start station latitude'], prediction.loc[prediction['ClusterType'] == 3, 'start station longitude'], s=33, c='cyan')
plt.scatter(prediction.loc[prediction['ClusterType'] == 4, 'start station latitude'], prediction.loc[prediction['ClusterType'] == 4, 'start station longitude'], s=33, c='magenta')
plt.title('Clusters of station locations')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

joblib.dump(finalkmeans, 'KMeans_Function.joblib')
