# -*- coding: utf-8 -*-
# ****************************************************************
# aitk.algorithms: Algorithms for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.algorithms
#
# ****************************************************************

import random
import numpy as np
import matplotlib.pyplot as plt

class NoveltySearch():
    """
    Novelty search is a method of searching for a variety of solutions
    rather than trying to optimize a particular goal. To do this, novelty
    search maintains an archive of behaviors, comparing current behaviors
    against the saved ones to determine whether a current behavior is novel
    enough to add to the archive. A current behavior's novelty is defined
    as the average distance to its k nearest neighbors in the archive.
    Behaviors are represented as lists of points.
    """
    def __init__(self, k, limit, threshold, max_dist):
        self.archive = []              # archive of past unique behaviors
        self.k = k                     # k nearest neighbors determine novelty
        self.limit = limit             # maximum size of the archive
        self.threshold = threshold     # how distant to be considered novel
        self.max_dist = max_dist       # max dist possible between two behaviors
        self.growth = []               # size of the archive over time

    def point_distance(self, p1, p2):
        """
        Returns the Euclidean distance between points p1 and p2, which are
        assumed to be lists of equal length. This method works for points of
        any length.
        """
        assert len(p1) == len(p2), "Lengths of points must be the same"
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.sqrt(np.sum( (a1-a2)**2 ))

    def behavior_distance(self, b1, b2):
        """
        Returns the sum of the point distances between behaviors b1 and b2,
        which are assumed to be lists of points of equal length. This method
        works for behaviors of any length.
        """
        assert len(b1) == len(b2), "Lengths of behaviors must the same"
        return sum([self.point_distance(p,q) for p, q in zip(b1, b2)])

    def k_nearest_distance(self, behavior):
        """
        For the given behavior, returns the sum of the behavior distances to
        its k nearest neighbors in the archive. If there are less than k
        behaviors in the archive, then returns the sum of all of the behaviors.
        """
        dists = [self.behavior_distance(behavior, novel_b[0])
                 for novel_b in self.archive]
        dists.sort()
        return sum(dists[:self.k])

    def sparseness(self, behavior):
        """
        For the given behavior, returns the normalized average distance
        (based on the max_dist) to its k nearest neighbors in the archive.
        If the archive is empty, then return 1.0. If the archive has less
        than k behaviors, the use the archive's current size to compute
        the average distance. The sparseness of a behavior will represent
        its novelty.
        """
        if len(self.archive) == 0:
            return 1.0
        d = self.k_nearest_distance(behavior)
        avg = d / min(self.k, len(self.archive))
        return avg / self.max_dist

    def check_archive(self, behavior, other_info=None):
        """
        Checks whether the given behavior is novel enough to add to the
        archive and returns its sparseness. All new behaviors should be added
        to the end of the archive. The behavior at the front of the archive
        is the oldest. When the size of the archive is less than k, then
        add the given behavior to the archive. Otherwise check if the
        sparseness of the given behavior is greater than the threshold, and
        if so, add the behavior to the archive. After adding a behavior, test
        if the archive size has exceeded the limit, and if so, remove the
        oldest behavior from the archive.
        """
        novelty = self.sparseness(behavior)
        if novelty > self.threshold or len(self.archive) < self.k:
            self.archive.append([behavior, novelty, other_info])
        if len(self.archive) > self.limit:
            self.archive.pop(0)
        self.growth.append(len(self.archive))
        return novelty

    def plot_growth(self):
        """
        Plot growth or archive size over time.
        """
        assert len(self.growth) > 0, "archive must be non-empty to plot growth"
        plt.title("Archive Growth")
        plt.xlabel("Time")
        plt.ylabel("Size")
        plt.plot(self.growth)
        plt.show()

