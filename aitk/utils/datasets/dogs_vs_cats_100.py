# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python AI utils
#
# Copyright (c) 2020 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

import os
import numpy as np
import glob
from PIL import Image

from .utils import get_file

_filename = get_file(
    "dogs-vs-cats-100.npz",
    "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/dogs-vs-cats/dogs-vs-cats-100.npz",
    extract=False,
)

def get_dogs_vs_cats_100():
    dataset = np.load(_filename)
    return dataset
