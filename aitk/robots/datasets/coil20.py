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
    "coil-20-no-background.zip",
    "https://media.githubusercontent.com/media/Calysto/conx-data/master/coil/coil-20-no-background.zip",
    extract=True,
)

DATA_DIR = os.path.splitext(_filename)[0]

DATA_RANGE = {
    "object-start": 1,
    "object-stop": 20,
    "slice-start": 0,
    "slice-stop": 720,
    "slice-increment": 10,
}


def get(obj_num, degree):
    offset = -180.0  # make 0 be facing forwards
    slice = degree * 2 + offset  # convert from 360 to 720
    slice = slice % 720  # map into range 0 to 720
    slice = max(
        min(round(round_to_nearest(slice, 10) / 10), 71), 0
    )  # get the nearest 1/10 unit
    filename = os.path.join(DATA_DIR, "obj%s__%s.png" % (obj_num, slice))
    image = Image.open(filename)
    return image
