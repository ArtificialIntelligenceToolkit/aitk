# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import numpy as np
from ipycanvas import Canvas
from ipywidgets import Layout

from .base import Backend


class CanvasBackend(Canvas, Backend):
    """
    Canvas Widget and a Jyrobot backend.
    """

    def __init__(self, *args, **kwargs):
        kwargs["layout"] = Layout(width="100%", height="auto")
        self.char_width = 4.8
        self.char_height = 10
        Canvas.__init__(self, *args, **kwargs)

    # aitk.robots API:

    def watch(self):
        # Return the Jupyter widget
        return self

    def is_async(self):
        # Does the backend take time to update the drawing?
        return True

    def get_dynamic_throttle(self, world):
        # A proxy to figure out how much to throttle
        return world._complexity * 0.005

    def get_image(self, time):
        """
        returns PIL.Image
        """
        from PIL import Image

        # self.image_data gives a PNG bytes
        # self.get_image_data() gives numpy array
        array = self.get_image_data()
        picture = Image.fromarray(array, "RGBA")
        return picture

    # High Level-API overloads:

    def draw_lines(self, points, stroke_style=None):
        if stroke_style:
            self.strokeStyle(stroke_style, 1)
        data = np.array(points)
        self.stroke_lines(data)

    def text(self, t, x, y):
        self.fill_text(t, x, y + self.char_height - 1)
