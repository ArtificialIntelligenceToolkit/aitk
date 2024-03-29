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

from .utils import get_file

_filename = get_file(
    "validata_6x6.data",
    "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/validate_6x6/validate_6x6.data",
    extract=True,
)

DATA_DIR = _filename

def onehot(num):
    retval = [0] * 10
    retval[num] = 1
    return retval

def get():
    """Each digit is represented by a 6x6 grid of 1's and 0's separated by blank lines."""
    filename = DATA_DIR
    fp = open(filename, "r")
    data = []
    while True:
        digit = []
        line = fp.readline()
        while len(line) != 0 and (line[0] == '0' or line[0] == '1'):
            line.strip()
            values = line.split()
            digit += [int(v) for v in values]
            line = fp.readline()
        if len(line) == 0:
            break
        if len(digit) > 0:
            data.append(digit)
    array = np.array(data)
    inputs = array.reshape((10, 6, 6))
    targets = np.array([onehot(v) for v in list(range(10))])
    return inputs, targets
