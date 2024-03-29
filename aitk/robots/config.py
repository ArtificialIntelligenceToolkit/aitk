# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************

import ast
import os

JYROBOTPATH = None
BACKEND = "pil"  # or any valid backends
ARGS = {}
VALID_BACKENDS = ["canvas", "svg", "debug", "pil"]


def get_aitk_search_paths():
    """
    Get the aitk.robots search paths
    """
    custom = os.environ.get("JYROBOTPATH", JYROBOTPATH)
    here = os.path.abspath(os.path.dirname(__file__))
    if custom is not None:
        if len(custom) > 0 and custom[-1] != "/":
            custom += "/"
        paths = [custom]
    else:
        paths = []
    paths += ["./", "./worlds/", os.path.join(here, "worlds/")]
    return paths


def set_aitk_path(path):
    """
    Set a custom search path for aitk.robots worlds
    """
    global JYROBOTPATH
    JYROBOTPATH = path


def setup_backend():
    global BACKEND, ARGS

    BACKEND = os.environ.get("JYROBOT_BACKEND", BACKEND)
    if ":" in BACKEND:
        BACKEND, ARGS = BACKEND.split(":", 1)
        ARGS = ast.literal_eval(ARGS)
    else:
        ARGS = {}


def switch_backend(backend=None, **kwargs):
    global BACKEND, ARGS

    if backend is None:
        return VALID_BACKENDS
    elif backend in VALID_BACKENDS:
        BACKEND = backend
        ARGS = kwargs
    else:
        raise ValueError("unknown backend type: %r" % backend)


def get_backend():
    return BACKEND, ARGS
