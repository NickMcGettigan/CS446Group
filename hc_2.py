# Marcus Kwong, Portland State University
# Adv ML
# hierarchical clustering experiment 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

GOOD_DATA = "stop_instances_102.csv"
BAD_DATA = "stop_instances_3179.csv"
UGLY_DATA = "stop_instances_1336.csv"
TEST_DATA = "wcd.csv"

STOP102 = (45.6343,	-122.531487)
STOP1336 = (45.684727, -122.660873)
STOP3179 = (45.531506, -122.656174)


#data = pd.read_csv(GOOD_DATA)
#data = pd.read_csv(BAD_DATA)
data = pd.read_csv(UGLY_DATA)
#data = pd.read_csv(TEST_DATA)
data.head()

#plt.figure(figsize=(10, 7))  
#plt.scatter(data['latitude'],data['longitude']) 
#plt.show()

"""
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled)
data_scaled.head()
"""

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.axhline(y=0.00015, color='r', linestyle='--')
#plt.show()

# apply HC for 2 clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)

# apply HC for 2 clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)

# visualize the 2 clusters
plt.figure(figsize=(10, 7))  
plt.scatter(data['latitude'], data['longitude'], c=cluster.labels_)

#plt.scatter(STOP102[0], STOP102[1])

plt.scatter(STOP1336[0], STOP1336[1])

#plt.scatter(STOP3179[0], STOP3179[1])

plt.show()