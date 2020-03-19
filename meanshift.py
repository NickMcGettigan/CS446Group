#!/usr/bin/python

# CS446 Group Project 
# Nick McGettigan

import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

CLUSTERS=3
FILENAME = "stop_instances_1336.csv"


def mean_cluster():
    X = numpy.loadtxt(open(FILENAME, "rb"), delimiter=',', skiprows=1)

    ms = MeanShift(bandwidth=0.001)
    ms.fit(X)
    # n_init = number of runs

    cluster_centers = ms.cluster_centers_
    print(cluster_centers)



def k_means():
    data = numpy.loadtxt(open(FILENAME, "rb"), delimiter=',', skiprows=1)
    #plt.scatter(data[:,0], data[:,1])
    #plt.savefig("test1.png")

    # n_init = number of runs
    km = KMeans(n_clusters=CLUSTERS, 
                init='random', 
                n_init=10, 
                max_iter=300, 
                tol=1e-04, 
                random_state=0)
    y_km = km.fit_predict(data)

    # for i in range(CLUSTERS):
    #     plt.scatter(
    #     data[y_km == i, 0], data[y_km == i, 1],
    #     s=50, c='lightgreen',
    #     marker='s', edgecolor='black',
    #     label='cluster ' + str(i)
    #     )


    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.ylim((-122.525,-122.550))
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig("test1.png")
    print(km.cluster_centers_)


if __name__ == "__main__":
    mean_cluster()
