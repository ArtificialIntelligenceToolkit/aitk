# -*- coding: utf-8 -*-
# **************************************************************
# aitk.keras: A Python Keras model API
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.keras
#
# **************************************************************

"""
Metrics can be computed as a stateless function:

metric(targets, outputs)

or as a stateful subclass of Metric.
"""

import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError

    @abstractmethod
    def update_state(self, targets, outputs):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

class ToleranceAccuracy(Metric):
    def __init__(self, tolerance):
        super().__init__("tolerance_accuracy")
        self.tolerance = tolerance
        self.reset_state()

    def reset_state(self):
        self.accurate = 0
        self.total = 0

    def update_state(self, targets, outputs):
        results = np.all(
            np.less_equal(np.abs(targets - outputs),
                          self.tolerance), axis=-1)
        self.accurate += sum(results)
        self.total += len(results)

    def result(self):
        return self.accurate / self.total

def tolerance_accuracy(targets, outputs):
    return np.mean(
        np.all(
            np.less_equal(np.abs(targets - outputs),
                          tolerance_accuracy.tolerance),
            axis=-1),
        axis=-1,
    )
# Needs the tolerance from somewhere:
tolerance_accuracy.tolerance = 0.1
