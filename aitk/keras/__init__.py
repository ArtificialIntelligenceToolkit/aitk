# -*- coding: utf-8 -*-
# **************************************************************
# aitk.keras: A Python Keras model API
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.keras
#
# **************************************************************

"""A module of basic building blcoks for constructing neural networks"""
from . import utils
from . import losses
from . import activations
from . import schedulers
from . import optimizers
from . import wrappers
from . import layers
from . import initializers
from . import modules
from . import models
from . import datasets

import sys
import numpy

# Create a fake module "backend" that is really numpy
backend = numpy
backend.image_data_format = lambda: 'channels_last'
sys.modules["aitk.keras.backend"] = backend
