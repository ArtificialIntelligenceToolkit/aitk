# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python utils for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

import os

from .utils import gallery, array_to_image, progress_bar, images_to_movie
from .joystick import Joystick, NoJoystick, has_ipywidgets, has_ipycanvas
from .joypad import JoyPad
from .datasets import get_dataset
from .colors import Color
from .grid import Grid

try:
    _in_colab = 'google.colab' in str(get_ipython())
except Exception:
    _in_colab = False

def in_colab():
    return _in_colab

def make_joystick(*args, **kwargs):
    """
    Make a joystick appropriate for your environment.

    Args:
        scale: the scaling of translate and rotate values; defaults (1.0, 1.0)
        width: width of the widget; default 250
        height: height of the widget; default 250
        function: the function to call when changing joystick; default print
    """
    if in_colab():
        return NoJoystick(*args, **kwargs)
    elif has_ipycanvas():
        return Joystick(*args, **kwargs)
    elif has_ipywidgets():
        return NoJoystick(*args, **kwargs)
    else:
        raise Exception("please install ipycanvas, or ipywidgets to use make_joystick")

def get_font(font_name):
    HERE = os.path.abspath(os.path.dirname(__file__))
    font_path = os.path.join(HERE, "fonts", font_name)
    if os.path.exists(font_path):
        return font_path
