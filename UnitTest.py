# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import unittest
import numpy as np
import EM
from Point import Point
from Cluster import Cluster


class MyTestCase(unittest.TestCase):
    global DATASET
    DATASET = "../dataSet/DSclustering/DS_3Clusters_999Points.txt"

    global point
    point = Point(np.array([2, 2]))

    global listPoints
    listPoints = [Point(np.array([1, 1])), Point(np.array([1, 3])),
                  Point(np.array([3, 1])), Point(np.array([3, 3]))]
    global cluster
    cluster = Cluster(listPoints, len(listPoints))

    # Check point dimension

    def test_dimension_point(self):
        self.assertEqual(point.dimension, 2)
        self.assertNotEquals(point.dimension, 1)

    # Check cluster dimension
    def test_dimension_cluster(self):
        self.assertEquals(cluster.dimension, 2)
        self.assertNotEquals(cluster.dimension, 3)

    # Check mean and  calculation
    def test_mean_std_cluster(self):
        mean = cluster.mean
        std = cluster.std
        self.assertEquals(mean[0], 2)
        self.assertEquals(mean[1], 2)
        self.assertEquals(std[0] - std[1], 0)

    # Check read data set file
    def test_read_file_points(self):
        points = EM.dataset_to_list_points(DATASET)
        self.assertTrue(len(points) > 0)
        self.assertTrue(points[0].dimension == 2)

    # Check probabilityCluster
    def test_get_probability_cluster(self):
        self.assertEquals(
            EM.get_probability_cluster(point, Cluster([point], 1)), 1)

    # Check cluster's method
    def test_cluster(self):
        cluster_test = Cluster([point], 1)
        self.assertEquals(cluster_test.dimension, 2)
        self.assertFalse(cluster_test.converge)
        np.testing.assert_array_equal(cluster_test.mean, np.array([2, 2]))
        np.testing.assert_array_equal(cluster_test.std, np.array([1, 1]))
        self.assertEquals(cluster_test.cluster_probability, 1)
        cluster_test.update_cluster(listPoints, 4)
        self.assertEquals(cluster_test.dimension, 2)
        self.assertTrue(cluster_test.converge)
        np.testing.assert_array_equal(cluster_test.mean, np.array([2, 2]))
        self.assertEquals(cluster_test.std[0] - cluster_test.std[1], 0)
        self.assertEquals(cluster_test.cluster_probability, 1)


if __name__ == '__main__':
    unittest.main()
