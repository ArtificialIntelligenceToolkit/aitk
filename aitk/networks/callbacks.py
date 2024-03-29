# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping


def match_acc(name):
    return (name.endswith("acc") or
            name.endswith("accuracy"))

def match_loss(name):
    return name == "loss"

def match_val(name):
    return name.startswith("val_")

def make_early_stop(monitor, patience):
    return EarlyStopping(monitor=monitor, patience=patience, verbose=True)

def make_stop(metric, goal, patience, use_validation):
    return StopWhen(metric, goal, patience, use_validation)

def make_save(network, save_rate):
    return SaveWeights(network, save_rate)

class UpdateCallback(Callback):
    def __init__(self, network, report_rate):
        super().__init__()
        self._network = network
        self._report_rate = report_rate
        self._figure = None

    def on_train_begin(self, logs=None):
        print("Training %s..." % self._network.name)

    def on_epoch_end(self, epoch, logs=None):
        self._network.on_epoch_end(self, logs, self._report_rate)

    def on_train_end(self, logs=None):
        self._network.on_epoch_end(self, logs)
        if self._figure is not None:
            plt.close()

class SaveWeights(Callback):
    def __init__(self, network, save_rate):
        super().__init__()
        self.network = network
        self.save_rate = save_rate

    def on_train_begin(self, logs=None):
        """
        Save the initial weights
        """
        if len(self.network._history["weights"]) == 0:
            self.network._history["weights"].append((0, self.network.get_weights()))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_rate == 0:
            self.network._history["weights"].append((epoch + 1, self.network.get_weights()))

class StopWhen(Callback):
    def __init__(self, metric="acc", goal=1.0, patience=0, use_validation=False, verbose=True):
        super().__init__()
        self.metric = metric
        self.goal = goal
        self.patience = patience
        self.use_validation = use_validation
        self.wait = 0
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.i_stopped_it = False

    def on_train_end(self, logs=None):
        if self.i_stopped_it and self.verbose > 0:
            prefix = "val_" if self.use_validation else ""
            item = "%s%s" % (prefix, self.metric)
            print("Stopped because %s beat goal of %s" % (item, self.goal))

    def compare(self, value, goal):
        # self.metric is either "accuracy" or "loss"
        if self.metric == "accuracy":
            return value >= goal
        else:
            return value <= goal

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            metric_value = None
            for key in logs:
                if self.use_validation:
                    if self.metric == "accuracy" and match_acc(key) and match_val(key):
                        metric_value = logs[key]
                        break
                    elif self.metric == "loss" and match_loss(key) and match_val(key):
                        metric_value = logs[key]
                        break
                else:
                    if self.metric == "accuracy" and match_acc(key) and not match_val(key):
                        metric_value = logs[key]
                        break
                    elif self.metric == "loss" and match_loss(key) and not match_val(key):
                        metric_value = logs[key]
                        break
            if metric_value is not None:
                if self.compare(metric_value, self.goal):
                    if self.wait >= self.patience:
                        self.model.stop_training = True
                        self.i_stopped_it = True
                    else:
                        self.wait += 1
                else:
                    # else, go back to zero and start over
                    self.wait = 0
