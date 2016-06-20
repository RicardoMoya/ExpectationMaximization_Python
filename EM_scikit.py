# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GMM

# Constant
DATASET1 = "./dataSet/DS_3Clusters_999Points.txt"
DATASET2 = "./dataSet/DS2_3Clusters_999Points.txt"
DATASET3 = "./dataSet/DS_5Clusters_10000Points.txt"
DATASET4 = "./dataSet/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
MAX_ITERATIONS = 10
CONVERGENCE_TOLERANCE = 0.001
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
          'cyan', 'magenta']


def dataset_to_list_points(dir_dataset):
    """
    Read a txt file with a set of points and return a list of objects Point
    :param dir_dataset:
    """
    points = list()
    with open(dir_dataset, 'rt') as reader:
        for point in reader:
            points.append(np.asarray(map(float, point.split("::"))))
    return points


def print_results(means_clusters, probability_clusters, label_cluster_points):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(means_clusters):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster %d' % label_cluster_points.count(i)
        print '\t\tCentroid: %s' % str(means_clusters[i])
        print '\t\tProbability: %02f%%' % (probability_clusters[i] * 100)


def plot_ellipse(center, covariance, alpha, color):
    """
    Plot the Ellipse that defines the area of Cluster
    :param center:
    :param covariance: covariance matrix
    :param alpha:
    :param color:
    :return: Ellipse
    """
    # eigenvalues and eigenvector of matrix covariance
    eigenvalues, eigenvector = np.linalg.eigh(covariance)
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


def plot_results(points, means_clusters, label_cluster_points,
                 covars_matrix_clusters):
    plt.plot()
    for nc in range(len(means_clusters)):
        # Plot points in cluster
        points_cluster = list()
        for i, p in enumerate(label_cluster_points):
            if p == nc:
                plt.plot(points[i][0], points[i][1], linestyle='None',
                         color=COLORS[nc], marker='.')
                points_cluster.append(points[i])
        # Plot mean
        mean = means_clusters[nc]
        plt.plot(mean[0], mean[1], 'o', markerfacecolor=COLORS[nc],
                 markeredgecolor='k', markersize=10)

        # Plot Ellipse
        plot_ellipse(mean, covars_matrix_clusters[nc], 0.2, COLORS[nc])

    plt.show()


def expectation_maximization(dataset, num_clusters, tolerance, max_iterations):
    # Read data set
    points = dataset_to_list_points(dataset)

    # Object GMM
    gmm = GMM(n_components=num_clusters, covariance_type='full', tol=tolerance,
              n_init=max_iterations, params='wmc')

    # Estimate Model (params='wmc'). Calculate, w=weights, m=mean, c=covars
    gmm.fit(points)

    # Predict Cluster of each point
    label_cluster_points = gmm.predict(points)

    means_clusters = gmm.means_
    probability_clusters = gmm.weights_
    covars_matrix_clusters = gmm.covars_

    # Print final result
    print_results(means_clusters, probability_clusters,
                  label_cluster_points.tolist())

    # Plot Final results
    plot_results(points, means_clusters, label_cluster_points,
                 covars_matrix_clusters)


if __name__ == '__main__':
    expectation_maximization(DATASET2, NUM_CLUSTERS, CONVERGENCE_TOLERANCE,
                             MAX_ITERATIONS)
