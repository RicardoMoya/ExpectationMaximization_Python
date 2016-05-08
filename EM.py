import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Point import Point
from Cluster import Cluster

# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

DATASET1 = "./dataSet/DS_3Clusters_999Points.txt"
DATASET2 = "./dataSet/DS2_3Clusters_999Points.txt"
DATASET3 = "./dataSet/DS_5Clusters_10000Points.txt"
DATASET4 = "./dataSet/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
ITERATIONS = 1000
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
          'cyan', 'magenta']


def dataset_to_list_points(dir_dataset):
    """
    Read a txt file with a set of points and return a list of objects Point
    :param dir_dataset: path file
    """
    points = list()
    with open(dir_dataset, 'rt') as reader:
        for point in reader:
            points.append(Point(np.asarray(map(float, point.split("::")))))
    return points


def get_probability_cluster(point, cluster):
    """
    Calculate the probability that the point belongs to the Cluster
    :param point:
    :param cluster:
    :return: probability =
    prob * SUM(e ^ (-1/2 * ((x(i) - mean)^2 / std(i)^2 )) / std(i))
    """
    mean = cluster.mean
    std = cluster.std
    prob = 1.0
    for i in range(point.dimension):
        prob *= (math.exp(-0.5 * (
            math.pow((point.coordinates[i] - mean[i]), 2) /
            math.pow(std[i], 2))) / std[i])

    return cluster.clusterProbability * prob


def get_expecation_cluster(clusters, point):
    """
    Returns the Cluster that has the highest probability of belonging to it
    :param clusters:
    :param point:
    :return: argmax (probability clusters)
    """
    expectation = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        expectation[i] = get_probability_cluster(point, c)

    return np.argmax(expectation)


def print_clusters_status(it_counter, clusters):
    print '\nITERATION %d' % it_counter
    for i, c in enumerate(clusters):
        print '\tCluster %d: Probability = %s; Mean = %s; Std = %s;' % (
            i + 1, str(c.clusterProbability), str(c.mean), str(c.std))


def print_results(clusters):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(clusters):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster: %d' % len(c.points)
        print '\t\tProbability: %s' % str(c.clusterProbability)
        print '\t\tMean: %s' % str(c.mean)
        print '\t\tStandard Desviation: %s' % str(c.std)


def plot_ellipse(center, points, alpha, color):
    """
    Plot the Ellipse that defines the area of Cluster
    :param center:
    :param points: points of cluster
    :param alpha:
    :param color:
    :return: Ellipse
    """

    # Matrix Covariance
    cov = np.cov(points, rowvar=False)

    # eigenvalues and eigenvector of matrix covariance
    eigenvalues, eigenvector = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvector = eigenvector[:, order]

    # Calculate Angle of ellipse
    angle = np.degrees(np.arctan2(*eigenvector[:, 0][::-1]))

    # Calculate with, height
    width, height = 4 * np.sqrt(eigenvalues[order])

    # Ellipse Object
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      alpha=alpha, color=color)

    ax = plt.gca()
    ax.add_artist(ellipse)

    return ellipse


def plot_results(clusters):
    plt.plot()
    for i, c in enumerate(clusters):
        # plot points
        x, y = zip(*[p.coordinates for p in c.points])
        plt.plot(x, y, linestyle='None', color=COLORS[i], marker='.')
        # plot centroids
        plt.plot(c.mean[0], c.mean[1], 'o', color=COLORS[i],
                 markeredgecolor='k', markersize=10)
        # plot area
        plot_ellipse(c.mean, [p.coordinates for p in c.points], 0.2, COLORS[i])

    plt.show()


def expectation_maximization(dataset, num_clusters, iterations):
    # Read data set
    points = dataset_to_list_points(dataset)

    # Select N points random to initiacize the N Clusters
    initial = random.sample(points, num_clusters)

    # Create N initial Clusters
    clusters = [Cluster([p], len(initial)) for p in initial]

    # Inicialize list of lists to save the new points of cluster
    new_points_cluster = [[] for i in range(num_clusters)]

    converge = False
    it_counter = 0
    while (not converge) and (it_counter < iterations):
        # Expectation Step
        for p in points:
            i_cluster = get_expecation_cluster(clusters, p)
            new_points_cluster[i_cluster].append(p)

        # Maximization Step
        for i, c in enumerate(clusters):
            c.update_cluster(new_points_cluster[i], len(points))

        # Check that converge all Clusters
        converge = [c.converge for c in clusters].count(False) == 0

        # Increment counter and delete lists of clusters points
        it_counter += 1
        new_points_cluster = [[] for i in range(num_clusters)]

        # Print clusters status
        print_clusters_status(it_counter, clusters)

    # Print final result
    print_results(clusters)

    # Plot Final results
    plot_results(clusters)


if __name__ == '__main__':
    expectation_maximization(DATASET1, NUM_CLUSTERS, ITERATIONS)
