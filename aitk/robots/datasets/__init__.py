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

_aitk_base_dir = os.path.expanduser("~")
if not os.access(_aitk_base_dir, os.W_OK):
    _aitk_base_dir = "/tmp"

_aitk_dir = os.path.join(_aitk_base_dir, ".aitk")

if not os.path.exists(_aitk_dir):
    try:
        os.makedirs(_aitk_dir)
    except OSError:
        pass


def get_dataset(dataset):
    get = None
    if dataset == "coil20":
        from .coil20 import get
    elif dataset == "coil100":
        from .coil100 import get
    return get
