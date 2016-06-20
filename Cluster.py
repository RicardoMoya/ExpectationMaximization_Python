# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np


class Cluster:
    """
    Class to represent a Cluster: set of points and their parameters (mean,
    standard desviation and probability of belonging to Cluster)
    """

    def __init__(self, points, total_points):
        if len(points) == 0:
            raise Exception("Cluster cannot have 0 Points")
        else:
            self.points = points
            self.dimension = points[0].dimension

        # Check that all elements of the cluster have the same dimension
        for p in points:
            if p.dimension != self.dimension:
                raise Exception(
                    "Point %s has dimension %d different with %d from the rest "
                    "of points") % (p, len(p), self.dimension)

        # Calculate mean, std and probability
        points_coordinates = [p.coordinates for p in self.points]
        self.mean = np.mean(points_coordinates, axis=0)
        self.std = np.array([1.0, 1.0])
        self.cluster_probability = len(self.points) / float(total_points)
        self.converge = False

    def update_cluster(self, points, total_points):
        """
        Calculate new parameters and check if converge (maximization step)
        :param total_points:
        :param points: list of new points
        :return: updated cluster
        """
        old_mean = self.mean
        self.points = points
        points_coordinates = [p.coordinates for p in self.points]
        self.mean = np.mean(points_coordinates, axis=0)
        self.std = np.std(points_coordinates, axis=0, ddof=1)
        self.cluster_probability = len(points) / float(total_points)
        self.converge = np.array_equal(old_mean, self.mean)

    def __repr__(self):
        cluster = 'Mean: ' + str(self.mean) + '\nDimension: ' + str(
            self.dimension)
        for p in self.points:
            cluster += '\n' + str(p)

        return cluster + '\n\n'
