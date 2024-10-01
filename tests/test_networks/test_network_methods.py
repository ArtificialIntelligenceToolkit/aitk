# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2024 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************


"""Questions:

In the file test_network.py, all test cases call connect
and compile before using the networks. However, the 
constructors for both the Network and SimpleNetwork classes
call self.compile. So is the compile call necessary?

Should the constructors also call self.connect by default. When would
there ever be a case when you wouldn't want the layers to be
connected?

"""


import numpy as np
from tensorflow.keras.layers import Dense, InputLayer
from aitk.networks import Network, SimpleNetwork
from aitk.utils import get_dataset


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


def test_propagate():
    network = SimpleNetwork(3, 2, 1)
    network.set_weights([1, 1, 1, 1, 1, 1, -2.5, -1.5, -3, 2, 0])
    inputs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    expected_activations = [
        [0.53426534],
        [0.5517651],
        [0.5280447],
        [0.44220227]
    ]
    for i in range(len(inputs)):
        result = network.propagate(np.array(inputs[i]))
        assert np.allclose(result, expected_activations[i])

        
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
    #     output
    #    /      \
    # hiddenA  hiddenB
    #    |        |
    # inputA   inputB
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
    network.compile()
    result = network.topological_sort(network._layers,
                                  network._get_input_layers())
    names = [layer.name for layer in result]
    assert names[0][:-1] == names[1][:-1] == "input"
    assert names[2][:-1] == names[3][:-1] == "hidden"
    assert names[4] == "output"

def test_get_input_from_dataset():
    network = SimpleNetwork(
        (6,6),
        "Flatten",
        10,
        (10, "softmax"))
    test_inputs, test_targets = get_dataset("validate_6x6")
    result = network.get_input_from_dataset(0, test_inputs)
    diff = result - test_inputs[0]
    assert np.count_nonzero(diff) == 0


def test_get_target_from_dataset():
    network = SimpleNetwork(
        (6,6),
        "Flatten",
        10,
        (10, "softmax"))
    test_inputs, test_targets = get_dataset("validate_6x6")
    result = network.get_target_from_dataset(0, test_targets)
    diff = result - test_targets[0]
    assert np.count_nonzero(diff) == 0

    
def test_get_input_from_banked_dataset():
    # outputA outputB
    #     \    /
    #     hidden
    #     /    \
    # inputA  inputB
    network = Network()
    network.add(InputLayer([2], name="inputA"))
    network.add(InputLayer([3], name="inputB"))
    network.add(Dense(4, name="hidden"))
    network.add(Dense(1, name="outputA"))
    network.add(Dense(2, name="outputB"))
    network.connect("inputA", "hidden")
    network.connect("inputB", "hidden")
    network.connect("hidden","outputA")
    network.connect("hidden","outputB")
    network.compile()
    inputs = [np.array([[0,0],[1,0],[1,1]]),
              np.array([[0,0,0],[1,0,1],[1,1,1]])]
    targets = [np.array([[0],[0],[1]]),
               np.array([[0,0],[1,0],[1,1]])]
    result = network.get_input_from_dataset(2, inputs)
    diff = inputs[0][2] - result[0]
    assert np.count_nonzero(diff) == 0
    diff = inputs[1][2] - result[1]
    assert np.count_nonzero(diff) == 0


def test_get_target_from_banked_dataset():
    # outputA outputB
    #     \    /
    #     hidden
    #     /    \
    # inputA  inputB
    network = Network()
    network.add(InputLayer([2], name="inputA"))
    network.add(InputLayer([3], name="inputB"))
    network.add(Dense(4, name="hidden"))
    network.add(Dense(1, name="outputA"))
    network.add(Dense(2, name="outputB"))
    network.connect("inputA", "hidden")
    network.connect("inputB", "hidden")
    network.connect("hidden","outputA")
    network.connect("hidden","outputB")
    network.compile()
    inputs = [np.array([[0,0],[1,0],[1,1]]),
              np.array([[0,0,0],[1,0,1],[1,1,1]])]
    targets = [np.array([[0],[0],[1]]),
               np.array([[0,0],[1,0],[1,1]])]
    result = network.get_target_from_dataset(2, inputs)
    diff = inputs[0][2] - result[0]
    assert np.count_nonzero(diff) == 0
    diff = inputs[1][2] - result[1]
    assert np.count_nonzero(diff) == 0
