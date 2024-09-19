# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2024 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

from tensorflow.keras.layers import Dense, InputLayer

from aitk.networks import Network, SimpleNetwork


def test_sample():
    network = SimpleNetwork(1, 2, 3, 2, 1)
    results = network.predict([1])

    assert len(results) == 1
