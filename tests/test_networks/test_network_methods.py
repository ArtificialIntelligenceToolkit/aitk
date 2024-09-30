# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2024 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************


"""
Questions:

In the file test_network.py, all test cases call connect
and compile before using the networks. However, the 
constructors for both the Network and SimpleNetwork classes
call self.compile. And I would argue that the constructors
should also call self.connect by default. When would there ever
be a case when you wouldn't want the layers to be connected?


"""


import numpy as np
from tensorflow.keras.layers import Dense, InputLayer

from aitk.networks import Network, SimpleNetwork


def test_set_weights():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    inputs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    expected_outputs = [
        [0.53426534],
        [0.5517651],
        [0.5280447],
        [0.44220227]
    ]
    for i in range(len(inputs)):
        output = network.propagate(inputs[i])
        assert np.allclose(output, expected_outputs[i])

def test_get_weights():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    # weights are returned in this order:
    # input->hidden
    # hidden->output
    # hidden biases
    # output biases
    weights = network.get_weights()
    assert len(weights[0]) == 3
    assert len(weights[1]) == 2
    assert len(weights[2]) == 2
    assert len(weights[3]) == 1
    
        
def test_propagate_to():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    inputs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    expected_activations = [
        [0.075858176, 0.18242551],
        [0.18242551, 0.37754068],
        [0.37754068, 0.62245935],
        [0.62245935, 0.8175745],
    ]
    for i in range(len(inputs)):
        actual_activations = list(network.propagate_to(inputs[i], "hidden"))
        assert np.allclose(actual_activations, expected_activations[i])

        
def test_predict():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    inputs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    expected_activations = [
        0.53426534,
        0.5517651,
        0.5280447,
        0.44220227
    ]
    results = network.predict(np.array(inputs))
    actual_activations = list(np.array(results).flatten())
    assert np.allclose(actual_activations, expected_activations)

    
def test_predict_to():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    inputs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    result = network.predict_to(np.array(inputs), "hidden")
    expected_activations = [
        [0.075858176, 0.18242551],
        [0.18242551, 0.37754068],
        [0.37754068, 0.62245935],
        [0.62245935, 0.8175745],
    ]
    for i in range(len(result)):
        actual_activations = list(result[i])
        assert np.allclose(list(result[i]), expected_activations[i])

        
def test_predict_from():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    expected_activations = [
        0.53426534,
        0.5517651,
        0.5280447,
        0.44220227
    ]
    hiddens = [
      [0.07585818, 0.18242551],
      [0.18242551, 0.37754068],
      [0.37754068, 0.62245935],
      [0.62245935, 0.8175745 ]
    ]
    results = network.predict_from(np.array(hiddens), "hidden", "output")
    actual_activations = list(np.array(results).flatten())
    assert np.allclose(actual_activations, expected_activations)


def test_train_from_set_weights():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    train_inputs = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    train_targets = [[0], [0], [0], [0], [1], [1], [1], [0]]
    history = network.fit(
        train_inputs,
        train_targets,
        batch_size=8,
        report_rate=100,
        epochs=1000,
        accuracy=1.0,
        tolerance=0.2,
    )
    assert len(history.history["tolerance_accuracy"]) == 874
    expected_weights = [
        2.348937,
        4.2549586,
        2.348937,
        4.2549586,
        2.348937,
        4.2549586,
        -5.95034,
        -5.579458,
        -6.8648214,
        7.5803447,
        -3.8168766,
    ]
    weights = network.get_weights()
    actual_weights = []
    for array in weights:
        actual_weights += list(array.flatten())
    assert np.allclose(expected_weights, actual_weights)

    
def test_topological_sort():
    network = Network()
    network.add(InputLayer([2], name="inputA"))
    network.add(InputLayer([3], name="inputB"))
    network.add(Dense(2, name="hiddenA"))
    network.add(Dense(3, name="hiddenB"))
    network.add(Dense(1, name="output"))
    network.connect("inputA", "hiddenA")
    network.connect("inputB", "hiddenB")
    network.connect("hiddenA","output")
    network.connect("hiddenB","output")
    result = network.topological_sort(network._layers,
                                  network._get_input_layers())
    names = [layer.name for layer in result]
    assert names[0][:-1] == names[1][:-1] == "input"
    assert names[2][:-1] == names[3][:-1] == "hidden"
    assert names[4] == "output"
    
