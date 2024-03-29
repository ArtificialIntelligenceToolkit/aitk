# -*- coding: utf-8 -*-
# **************************************************************
# aitk.keras: A Python Keras model API
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.keras
#
# **************************************************************

from ..layers import Input, Activation, Concatenate
from ..losses import MeanSquaredError, CrossEntropy
from ..initializers import OptimizerInitializer
from ..callbacks import History
from ..utils import topological_sort

import numpy as np
import time
import math
import numbers
import functools
import operator
from collections import defaultdict

LOSS_FUNCTIONS = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "crossentropy": CrossEntropy,
    # FIXME: add more error functions
}

NAME_CACHE = {}

def get_metric_name(metric):
    if hasattr(metric, "name"):
        return metric.name
    elif hasattr(metric, "__name__"):
        return metric.__name__
    else:
        return str(metric)
        

class Model():
    def __init__(self, inputs=None, outputs=None, name=None):
        self.stop_training = False
        self.built = False
        self.sequential = False
        self.history = History()
        self.name = self.make_name(name)
        self.layers = []
        self.layer_map = {}
        self._input_layers = None
        self._output_layers = None
        self.step = 0
        # Build a model graph from inputs to outputs:
        if inputs is not None and outputs is not None:
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            queue = [] if inputs is None else inputs
            if not isinstance(queue, (list, tuple)):
                queue = [queue]
            while len(queue) > 0:
                layer = queue.pop(0)
                if layer not in self.layers:
                    if layer.name in self.layer_map:
                        raise AttributeError("duplicate layer name: '%s'" % layer.name)
                    self.layers.append(layer)
                    self.layer_map[layer.name] = layer
                if layer in outputs:
                    # Make sure no more layers:
                    layer.output_layers = []
                else:
                    queue.extend(layer.output_layers)
            self.sequential = self.is_sequential()
            self.build()

    def is_sequential(self):
        return ((len(self.get_input_layers()) == 1) and
                (len(self.get_output_layers()) == 1) and
                (not any([isinstance(layer, Concatenate)
                          for layer in self.layers])))

    def get_input_layers(self):
        if self._input_layers is None:
            return [layer for layer in self.layers if len(layer.input_layers) == 0]
        else:
            return self._input_layers

    def get_output_layers(self):
        if self._output_layers is None:
            return [layer for layer in self.layers if len(layer.output_layers) == 0]
        else:
            return self._output_layers

    def connect(self, in_layer, out_layer):
        """
        Connect first layer to second layer.
        """
        if in_layer not in out_layer.input_layers:
            out_layer.input_layers.append(in_layer)
        if out_layer not in in_layer.output_layers:
            in_layer.output_layers.append(out_layer)

    def make_name(self, name):
        if name is None:
            class_name = self.__class__.__name__.lower()
            count = NAME_CACHE.get(class_name, 0)
            if count == 0:
                new_name = class_name
            else:
                new_name = "%s_%s" % (class_name, count)
            NAME_CACHE[class_name] = count + 1
            return new_name
        else:
            return name

    def summary(self):
        if not self.built:
            print(f'Model: "{self.name}" (unbuilt)')
        else:
            print(f'Model: "{self.name}"')
        print('_' * 65)
        print("Layer (type)                 Output Shape              Param #")
        print("=" * 65)
        total_parameters = 0
        # FIXME: sum up other, non-trainable params
        other_params = 0
        for i, layer in enumerate(topological_sort(self.get_input_layers())):
            layer_name = ("%s (%s)" % (layer.name, layer.__class__.__name__))[:25]
            output_shape = (None, layer.n_out) if isinstance(layer.n_out, numbers.Number) else layer.n_out
            if self.built:
                parameters = sum([np.prod(item.shape) for item in layer.parameters.values() if item is not None])
                total_parameters += parameters
                print(f"{layer_name:25s} {str(output_shape)[:15]:>15s} {parameters:>20,}")
            else:
                print(f"{layer_name:25s} {str(output_shape)[:15]:>15s} {'(unbuilt)':>20}")
            if i != len(self.layers) - 1:
                print("_" * 65)
        print("=" * 65)
        if self.built:
            print(f"Total params: {total_parameters:,}")
            print(f"Trainable params: {total_parameters + other_params:,}")
            print(f"Non-trainable params: {other_params:,}")
        print("_" * 65)

    def build(self):
        self._input_layers = [layer for layer in self.layers if len(layer.input_layers) == 0]
        self._output_layers = [layer for layer in self.layers if len(layer.output_layers) == 0]
        for layer in self.layers:
            if not isinstance(layer, Input):
                self.is_initialized = False
        # now, let's force the layers to initialize:
        inputs = self.build_inputs()
        self.predict(inputs)
        self.built = True

    def compile(self, optimizer, loss, metrics=None):
        for layer in self.layers:
            if not isinstance(layer, Input):
                self.is_initialized = False
                layer.optimizer = OptimizerInitializer(optimizer)()
                loss_function = LOSS_FUNCTIONS[loss]
                self.loss_function = loss_function()
        self.metrics = metrics if metrics is not None else []
        self.build()

    def get_layer_output_shape(self, layer, n=1):
        """
        Get the shape of the layer with a dataset
        size of n.
        """
        if isinstance(layer.n_out, numbers.Number):
            shape = (n, layer.n_out)
        else:
            shape = tuple([n] + list(layer.n_out))
        return shape

    def get_layer_output_array(self, layer):
        """
        Get an output array of a layer (dataset, n = 1).
        """
        shape = self.get_layer_output_shape(layer)
        output = np.ndarray(shape)
        return output

    def build_inputs(self):
        """
        Build a dataset of dummy inputs.
        """
        if self.sequential:
            inputs = self.get_layer_output_array(self.layers[0])
        else:
            if len(self.get_input_layers()) > 1:
                inputs = [self.get_layer_output_array(input)
                          for input in self._input_layers]
            else:
                inputs = self.get_layer_output_array(self._input_layers[0])
        return inputs

    def get_weights(self, flat=False):
        """
        Get the weights from the model.
        """
        array = []
        if flat:
            for layer in self.layers:
                if layer.has_trainable_params():
                    for weight in layer.get_weights():
                        if isinstance(weight, numbers.Number):
                            array.extend(weight)
                        else:
                            array.extend(weight.flatten())
        else:
            for layer in self.layers:
                if layer.has_trainable_params():
                    array.extend(layer.get_weights())
        return array

    def copy_weights(self, model):
        """
        Copy the weights from another model by layer name.
        """
        for layer in model.layers:
            weights = layer.get_weights()
            self.layer_map[layer.name].set_weights(weights)

    def get_weights_by_name(self):
        """
        Copy the weights from another model by layer name.
        """
        return {layer.name: layer.get_weights() for layer in self.layers}

    def set_weights(self, weights):
        """
        Set the weights in a network.

        Args:
            weights: a list of pairs of weights and biases for each layer,
                or a single (flat) array of values
        """
        if len(weights) > 0 and isinstance(weights[0], numbers.Number):
            # Flat
            current = 0
            for layer in self.layers:
                if layer.has_trainable_params():
                    orig = layer.get_weights()
                    new_weights = []
                    for item in orig:
                        if isinstance(item, numbers.Number):
                            total = 1
                            new_weights.append(item)
                        else:
                            total = functools.reduce(operator.mul, item.shape, 1)
                            w = np.array(weights[current:current + total], dtype=float)
                            new_weights.append(w.reshape(item.shape))
                        current += total
                    layer.set_weights(new_weights)
        else:
            i = 0
            for layer in self.layers:
                if layer.has_trainable_params():
                    orig = layer.get_weights()
                    count = len(orig)
                    layer.set_weights(weights[i:i+count])
                    i += count

    def format_time(self, seconds):
        """
        Format time for easy human reading.
        """
        if seconds > 1:
            return f"{seconds:.0f}s"
        elif seconds * 1000 > 1:
            return f"{seconds * 1000:.0f}ms"
        else:
            return f"{seconds * 1000000:.0f}µs"

    def fit(self, inputs, targets, batch_size=32, epochs=1, verbose="auto", callbacks=None,
            initial_epoch=0, shuffle=True):
        """
        The training loop for all models.
        """
        self.history = History()
        self.stop_training = False
        verbose = 1 if verbose == "auto" else verbose
        callbacks = [] if callbacks is None else callbacks
        callbacks.append(self.history)
        inputs = np.array(inputs, dtype=float)
        targets = np.array(targets, dtype=float)
        self.flush_gradients()
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()
        for epoch in range(initial_epoch, epochs):
            if self.stop_training:
                break
            epoch_metric_values = {}
            for metric in self.metrics:
                if hasattr(metric, "reset_state"):
                    metric.reset_state()
                else:
                    epoch_metric_values[get_metric_name(metric)] = 0

            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            loss = 0
            total_batches = math.ceil(self.get_length_of_inputs(inputs) / batch_size)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            for batch, length, batch_data in self.enumerate_batches(inputs, targets, batch_size, shuffle):
                start_time = time.monotonic()
                batch_loss, batch_metric_values = self.train_batch(batch_data, batch, length, batch_size, callbacks)
                loss += batch_loss
                for metric in batch_metric_values:
                    # FIXME: Need to account for uneven batch sizes?
                    epoch_metric_values[metric] += batch_metric_values[metric]
                end_time = time.monotonic()
                self.step += length
                if verbose:
                    logs = {}
                    ftime = self.format_time((end_time - start_time) / length)
                    for metric in self.metrics:
                        if hasattr(metric, "result"):
                            logs[metric.name] = metric.result()
                        else:
                            if get_metric_name(metric) in batch_metric_values:
                                logs[get_metric_name(metric)] = batch_metric_values[get_metric_name(metric)]
                    metrics = " - ".join(["%s: %.4f" % (metric, logs[metric]) for metric in batch_metric_values])
                    if metrics:
                        metrics = " - " + metrics
                # ideally update output here
            logs = {
                "loss": loss,
            }
            for metric in self.metrics:
                if hasattr(metric, "result"):
                    logs[metric.name] = metric.result()
                else:
                    if get_metric_name(metric) in epoch_metric_values:
                        logs[get_metric_name(metric)] = epoch_metric_values[get_metric_name(metric)] / total_batches
            if verbose:
                metrics = " - ".join(["%s: %.4f" % (metric, logs[metric]) for metric in logs])
                if metrics:
                        metrics = " - " + metrics
                # Until we have output screen formatting; uses the last computed times, metrics
                print(f"{batch + 1}/{total_batches} [==============================] - {end_time - start_time:.0f}s {ftime}/step{metrics}")
            for callback in callbacks:
                callback.on_epoch_end(
                    epoch,
                    logs
                )
        if self.stop_training:
            print("Training stopped early.")
        for callback in callbacks:
            callback.on_train_end()
        return self.history

    def flush_gradients(self):
        for layer in self.layers:
            if layer.has_trainable_params():
                layer.flush_gradients()

    def enumerate_batches(self, inputs, targets, batch_size, shuffle):
        indexes = np.arange(self.get_length_of_inputs(inputs))
        if shuffle:
            # In place shuffle
            np.random.shuffle(indexes)
        current_row = 0
        batch = 0
        while (current_row * batch_size) < self.get_length_of_inputs(inputs):
            batch_inputs = self.get_batch_inputs(
                inputs, indexes, current_row, batch_size)
            batch_targets = self.get_batch_targets(
                targets, indexes, current_row, batch_size)
            current_row += 1
            yield batch, self.get_length_of_inputs(batch_inputs), (batch_inputs, batch_targets)
            batch += 1

    def get_length_of_inputs(self, inputs):
        if len(self.get_input_layers()) == 1:
            return len(inputs)
        else:
            return len(inputs[0])

    def get_batch_inputs(self, inputs, indexes, current_row, batch_size):
        batch_indexes = indexes[current_row:current_row + batch_size]
        if len(self.get_input_layers()) == 1:
            return inputs[batch_indexes]
        else:
            return [np.array(inputs[i][batch_indexes])
                    for i in range(len(self.get_input_layers()))]

    def get_batch_targets(self, targets, indexes, current_row, batch_size):
        batch_indexes = indexes[current_row:current_row + batch_size]
        if self.sequential:
            # Numpy, one bank:
            return targets[batch_indexes]
        else:
            return [np.array(targets[i][batch_indexes])
                    for i in range(len(self.get_output_layers()))]

    def train_batch(self, dataset, batch, length, batch_size, callbacks):
        """
        dataset = (inputs, targets)
        batch = batch number (eg, step)
        length = the actual size of the batch
        batch_size = desired size of batch
        """
        inputs, targets = dataset
        # If the size of this batch is less than desired, scale it?
        #scale = length / batch_size
        scale = 1.0
        # Use predict to forward the activations, saving
        # needed information:
        outputs = self.predict(inputs, True)
        # Compute the derivative with respect
        # to this batch of the dataset:
        batch_loss = 0
        batch_metric_values = defaultdict(int)
        for callback in callbacks:
            callback.on_train_batch_begin(batch)
        results = 0
        # FIXME: If batch_size is different from others? Scale it?
        if self.sequential:
            dY_pred = self.loss_function.grad(
                targets,
                outputs,
            )
            queue = [(self.get_output_layers()[0], dY_pred)]
            while len(queue) > 0:
                layer, dY_pred = queue.pop(0)
                if not isinstance(layer, Input):
                    dY_pred = layer.backward(dY_pred)
                    for input_layer in layer.input_layers:
                        queue.append((input_layer, dY_pred))

            batch_loss = self.loss_function(targets, outputs) * scale
            for metric in self.metrics:
                if hasattr(metric, "update_state"):
                    metric.update_state(targets, outputs)
                else:
                    batch_metric_values[get_metric_name(metric)] = metric(targets, outputs)
        else:
            for out_n in range(len(self.get_output_layers())):
                dY_pred = self.loss_function.grad(
                    targets[out_n],
                    outputs[out_n],
                ) * scale
                queue = [(self.get_output_layers()[out_n], dY_pred)]
                while len(queue) > 0:
                    layer, dY_pred = queue.pop(0)
                    if not isinstance(layer, Input):
                        dY_pred = layer.backward(dY_pred)
                        for input_layer in layer.input_layers:
                            queue.append((input_layer, dY_pred))

                batch_loss += self.loss_function(targets[out_n], outputs[out_n]) * scale
                for metric in self.metrics:
                    if hasattr(metric, "update_state"):
                        metric.update_state(targets[out_n], outputs[out_n])
                    else:
                        batch_metric_values[get_metric_name(metric)] += metric(targets, outputs)

        for callback in callbacks:
            logs = {"batch_loss": batch_loss}
            logs.update(batch_metric_values)
            callback.on_train_batch_end(batch, logs)
        self.update(batch_loss)
        return batch_loss, batch_metric_values

    def update(self, batch_loss):
        """
        Update the weights based on the batch_loss.
        The weight delatas were computed in train_batch().
        """
        # FIXME? Need to pass the batch_loss to just the layers
        # responsible for this loss (eg, in case of multiple
        # output layers)
        # FIXME: layers need to be able to accumulate delta changes
        for layer in self.layers:
            if not isinstance(layer, Input):
                layer.update(batch_loss)

    def predict(self, inputs, retain_derived=False):
        inputs = np.array(inputs, dtype=float)
        results = []
        # First, load the outputs of the input layers:
        if self.sequential:
            outputs = {self._input_layers[0].name: inputs}
        else:
            if len(self._input_layers) > 1:
                outputs = {self._input_layers[i].name: input for i, input in enumerate(inputs)}
            else:
                outputs = {self._input_layers[0].name: inputs}

        # Propagate in topological order:
        for layer in topological_sort(self.get_input_layers()):
            if not isinstance(layer, Input):
                inputs = [outputs[in_layer.name] for in_layer in layer.input_layers]
                if len(inputs) == 1:
                    outputs[layer.name] = layer.forward(inputs[0], retain_derived=retain_derived)
                else:
                    outputs[layer.name] = layer.forward(inputs, retain_derived=retain_derived)

        for layer in self.get_output_layers():
            results.append(outputs[layer.name])
        if self.sequential:
            return results[0]
        else:
            return results

class Sequential(Model):
    def __init__(self, layers=None, name="sequential"):
        super().__init__(name=name)
        self.sequential = True
        if layers is not None:
            for layer in layers:
                self.add(layer)
            self.build()

    def add(self, layer):
        if layer.name in self.layer_map:
            raise AttributeError("duplicate layer name: '%s'" % layer.name)
        self.layer_map[layer.name] = layer
        if len(self.layers) == 0:
            if isinstance(layer, Input):
                self.layers.append(layer)
            else:
                input_layer = Input(input_shape=layer.input_shape)
                self.connect(input_layer, layer)
                self.layers.append(input_layer)
                self.layers.append(layer)
        elif isinstance(layer, Activation):
            self.layers[-1].act_fn = layer.activation
        else:
            input_layer = self.layers[-1]
            self.connect(input_layer, layer)
            self.layers.append(layer)
        self.build()
