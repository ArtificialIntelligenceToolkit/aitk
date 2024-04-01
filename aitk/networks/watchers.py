# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

from matplotlib import cm
import numpy as np
import re

from .utils import (
    image_to_uri,
)

class WeightWatcher():
    def __init__(self, network, to_name):
        self.network = network
        self.to_name = to_name
        self.to_layer = self.network[self.to_name]
        self.from_layer = self.to_layer.inbound_nodes[0].inbound_layers
        self.name = "WeightWatcher: to %s" % (self.to_name,)
        self._widget = None
        self._widget = self.get_widget()

    def array_to_image(self, vector):
        from PIL import Image, ImageDraw

        size = 1 # self.config.get("pixels_per_unit", 1)
        new_width = vector.shape[0] * size  # in, pixels
        new_height = vector.shape[1] * size  # in, pixels

        #try:
        #    cm_hot = cm.get_cmap(color)
        #except Exception:
        cm_hot = cm.get_cmap("gray")

        vector = cm_hot(vector)
        vector = np.uint8(vector * 255)
        if max(vector.shape) <= 20: # self.config["max_draw_units"]:
            # Need to make it bigger, to draw circles:
            # Make this value too small, and borders are blocky;
            # too big and borders are too thin
            scale = int(250 / max(vector.shape))
            size = size * scale
            image = Image.new(
                "RGBA", (new_height * scale, new_width * scale), color="white"
            )
            draw = ImageDraw.Draw(image)
            for row in range(vector.shape[1]):
                for col in range(vector.shape[0]):
                    # upper-left, lower-right:
                    draw.rectangle(
                        (
                            row * size,
                            col * size,
                            (row + 1) * size - 1,
                            (col + 1) * size - 1,
                        ),
                        fill=tuple(vector[col][row]),
                        outline="black",
                    )
        else:
            image = Image.fromarray(array)
            image = image.resize((new_height, new_width))

        return image

    def update(self, *args, **kwargs):
        weights = self.to_layer.get_weights() # matrix, bias
        # if len() == 2, weights and then biases
        # weights.shape = (incoming units, outgoing units)
        image = self.array_to_image(weights[0]) # weights
        image_uri = image_to_uri(image)
        width, height = image.size
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
        self._widget.value = div

    def get_widget(self):
        from ipywidgets import HTML

        if self._widget is None:
            self._widget = HTML()

        self.update()

        return self._widget

class LayerWatcher():
    def __init__(self, network, layer_name):
        self.name = "LayerWatcher: %s" % (layer_name)
        self.network = network
        self.layer_name = layer_name
        self._widget = None
        self.get_widget()

    def update(self, inputs=None, targets=None):
        if inputs is None and targets is None:
            return

        if len(self.network.input_bank_order) == 1:
            inputs = [np.array([inputs])]
        else:
            inputs = [np.array([bank]) for bank in inputs]

        image = self.network.make_image(
            self.layer_name, self.network.predict_to(inputs, self.layer_name)[0]
        )
        image_uri = image_to_uri(image)
        width, height = image.size
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
        self._widget.value = div

    def get_widget(self):
        from ipywidgets import HTML

        if self._widget is None:
            self._widget = HTML()
            image = self.network.make_image(
                self.layer_name, np.array(self.network.make_dummy_vector(self.layer_name)),
            )
            image_uri = image_to_uri(image)
            width, height = image.size
            div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
            self._widget.value = div

        return self._widget


class NetworkWatcher():
    def __init__(self,
                 network,
                 show_error=None,
                 show_targets=None,
                 rotate=None,
                 scale=None,
    ):
        self.name = "NetworkWatcher"
        self.network = network
        self._widget_kwargs = {}
        self._widget = None
        # Update the defaults:
        if show_error is not None:
            self._widget_kwargs["show_error"] = show_error
        if show_targets is not None:
            self._widget_kwargs["show_targets"] = show_targets
        if rotate is not None:
            self._widget_kwargs["rotate"] = rotate
        if scale is not None:
            self._widget_kwargs["scale"] = scale
        self.get_widget(**self._widget_kwargs)

    def update(self, inputs=None, targets=None):
        if inputs is None and targets is None:
            return

        svg = self.network.get_image(inputs, targets, return_type="svg", **self._widget_kwargs)

        # Watched items get a border
        # Need width and height; we get it out of svg:
        header = svg.split("\n")[0]
        width = int(re.match('.*width="(\d*)px"', header).groups()[0])
        height = int(re.match('.*height="(\d*)px"', header).groups()[0])
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;">%s</div>""" % (width, height, svg)
        self._widget.value = div

    def get_widget(self,
        show_error=None,
        show_targets=None,
        rotate=None,
        scale=None,
    ):
        """
        """
        from ipywidgets import HTML

        # Update the defaults:
        if show_error is not None:
            self._widget_kwargs["show_error"] = show_error
        if show_targets is not None:
            self._widget_kwargs["show_targets"] = show_targets
        if rotate is not None:
            self._widget_kwargs["rotate"] = rotate
        if scale is not None:
            self._widget_kwargs["scale"] = scale

        svg = self.network.get_image(return_type="svg", **self._widget_kwargs)

        # Watched items get a border
        # Need width and height; we get it out of svg:
        header = svg.split("\n")[0]
        width = int(re.match('.*width="(\d*)px"', header).groups()[0])
        height = int(re.match('.*height="(\d*)px"', header).groups()[0])
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;">%s</div>""" % (width, height, svg)

        if self._widget is None:
            # Singleton:
            self._widget = HTML(value=div)
        else:
            self._widget.value = div

        return self._widget
