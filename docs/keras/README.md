aitk.keras
==========

An implementation of the main Keras API with the layers in numpy.

*UNDER DEVELOPMENT*

Why?

* useful to explain deep learning
* can be used where tensorflow is not available (eg, JupterLite)

## Features

* supports Keras's Sequential and functional APIs
* alternative dataset downloader for JupyterLite

Examples:

```python
# Classic XOR
from aitk.keras.layers import Input, Dense
from aitk.keras.models import Sequential

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

model = Sequential()
model.add(Input(2, name="input"))
model.add(Dense(8, activation="tanh", name="hidden"))
model.add(Dense(1, activation="sigmoid", name="output"))
model.compile(optimizer="adam", loss="mse")

outputs = model.predict(inputs)
model.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)
```

See the notebook directory for additional examples.

See also the examples in the tests folder.

## Development

* implement shuffle
* report metrics to logs/history
* probably lots of edge cases ar broken
* see "FIXME" items in code

To run the tests:

```
$ pytest -vvv tests
```

Please feel free to report issues and make Pull Requests!

## References

Lowlevel numpy code based on [numpy_ml](https://github.com/ddbourgin/numpy-ml).
