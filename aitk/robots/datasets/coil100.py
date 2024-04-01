# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************

import os

from PIL import Image

from .utils import get_file, round_to_nearest

_filename = get_file(
    "coil-100-no-background.zip",
    "https://media.githubusercontent.com/media/Calysto/conx-data/master/coil/coil-100-no-background.zip",
    extract=True,
)

DATA_DIR = os.path.splitext(_filename)[0]


def get_range():
    # start, stop, increment
    return {
        "object-start": 1,
        "object-stop": 100,
        "slice-start": 0,
        "slice-stop": 355,
        "slice-increment": 5,
    }


def get(obj_num, degree):
    slice = degree % 360  # map into range 0 to 360
    slice = min(max(round_to_nearest(slice, 5), 0), 355)  # get the nearest 5th degree
    filename = os.path.join(DATA_DIR, "obj%s__%s.png" % (obj_num, slice))
    image = Image.open(filename)
    return image
