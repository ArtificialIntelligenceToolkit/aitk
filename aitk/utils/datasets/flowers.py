# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python AI utils
#
# Copyright (c) 2020 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

import numpy as np
import PIL.Image
import os

from .utils import get_file

_filename = get_file(
    "flowers-100.zip",
    "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/flowers/flowers-100.zip",
    extract=True,
)

DATA_DIR = os.path.splitext(_filename)[0]

def get():
    data = []
    for i in range(100):
        image = PIL.Image.open(os.path.join(DATA_DIR, "Image_%s.jpg" % i))
        data.append(np.array(image))
    return data
