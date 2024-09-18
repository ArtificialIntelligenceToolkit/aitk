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


def test_network_names():
    network = Network()
    network.add(InputLayer([1]))
    network.add(InputLayer([2]))
    network.add(Dense(5))
    network.add(Dense(6))

    assert network._layers[0].name.startswith("input")
    assert network._layers[1].name.startswith("input_")
    assert network._layers[2].name.startswith("dense")
    assert network._layers[3].name.startswith("dense_")


def test_network_names_again():
    # Should still follow this pattern
    network = Network()
    network.add(InputLayer([1]))
    network.add(InputLayer([2]))
    network.add(Dense(5))
    network.add(Dense(6))

    assert network._layers[0].name.startswith("input")
    assert network._layers[1].name.startswith("input_")
    assert network._layers[2].name.startswith("dense")
    assert network._layers[3].name.startswith("dense_")


def test_network_sequential_1():
    network = Network()
    network.add(InputLayer([2]))
    network.add(Dense(5))
    network.add(Dense(10))

    network.connect()
    network.compile()

    output = network.predict([[1, 1]])

    assert len(output) == 1
    assert len(output[0]) == 10


def test_network_sequential_2():
    network = SimpleNetwork(
        InputLayer([2]),
        Dense(5),
        Dense(10),
    )

    network.connect()
    network.compile()

    output = network.propagate([1, 1])

    assert len(output) == 10


def test_network_sequential_3():
    network = SimpleNetwork(
        [2],
        5,
        10,
    )

    network.connect()
    network.compile()

    output = network.propagate([1, 1])

    assert len(output) == 10


def test_network_sequential_4():
    network = SimpleNetwork(
        2,
        5,
        10,
    )

    network.connect()
    network.compile()

    output = network.propagate([1, 1])

    assert len(output) == 10


def test_network_display():
    network = SimpleNetwork(
        2,
        5,
        10,
    )

    network.connect()
    network.compile()

    output = network.display([1, 1], return_type="image")

    assert output.size == (400, 260)


def test_network_multi_inputs():
    network = Network()
    network.add(InputLayer([1], name="input-1"))
    network.add(InputLayer([2], name="input-2"))
    network.add(Dense(5, name="hidden"))
    network.add(Dense(6, name="output"))

    network.connect("input-1", "hidden")
    network.connect("input-2", "hidden")
    network.connect("hidden", "output")

    network.compile()

    output = network.propagate([[1], [1, 2]])

    assert len(output) == 6


def test_network_multi_outputs():
    network = Network()
    network.add(InputLayer([1], name="input-1"))
    network.add(Dense(5, name="hidden"))
    network.add(Dense(2, name="output-1"))
    network.add(Dense(3, name="output-2"))

    network.connect("input-1", "hidden")
    network.connect("hidden", "output-1")
    network.connect("hidden", "output-2")

    network.compile()

    output = network.propagate([1])

    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[1]) == 3


def test_network_multi_inputs_outputs():
    network = Network()
    network.add(InputLayer([1], name="input-1"))
    network.add(InputLayer([2], name="input-2"))
    network.add(Dense(5, name="hidden"))
    network.add(Dense(2, name="output-1"))
    network.add(Dense(3, name="output-2"))

    network.connect("input-1", "hidden")
    network.connect("input-2", "hidden")
    network.connect("hidden", "output-1")
    network.connect("hidden", "output-2")

    network.compile()

    output = network.propagate([[1], [0, 0.5]])

    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[1]) == 3
