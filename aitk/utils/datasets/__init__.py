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


def get_dataset(dataset=None):
    """
    Download and return a dataset.

    Args:
        dataset (str): one of "digits6x6", "dogs-vs-cats",
            "dogs", "cats", "dogs-vs-cats-100", "gridfonts",
            "figure-ground-a", "cmu-faces", "cmu-faces-full",
            "cmu-faces-half", "cmu-faces-quarter",
            "validate_6x6", "nanoGPT_shakespeare", "flowers",

    Examples:
    ```python
    >>> get_dataset()
    ["cats", "digits6x6", "dogs", "dogs-vs-cats", "dogs-vs-cats-100",
     "gridfonts", "figure-ground-a", "cmu-faces", "cmu-faces-full",
     "cmu-faces-half", "cmu-faces-quarter", "validate_6x6",
     "nanoGPT_shakespeare", "flowers"]

    >>> dataset = get_dataset("dogs")
    ```
    """
    if dataset is None:
        return [
            "cats",
            "digits6x6",
            "dogs",
            "dogs-vs-cats",
            "dogs-vs-cats-100",
            "gridfonts",
            "flowers",
            "figure-ground-a",
            "validate_6x6",
            "cmu-faces",
            "cmu-faces-full",
            "cmu-faces-half",
            "cmu-faces-quarter",
            "nanoGPT_shakespeare",
        ]
    get = None
    if dataset == "digits6x6":
        from .digits6x6 import get
    elif dataset == "validate_6x6":
        from .validate_6x6 import get
    elif dataset == "dogs-vs-cats":
        from .dogs_vs_cats import get
    elif dataset == "dogs":
        from .dogs_vs_cats import get_dogs as get
    elif dataset == "cats":
        from .dogs_vs_cats import get_cats as get
    elif dataset == "dogs":
        from .dogs_vs_cats import get_dogs as get
    elif dataset == "dogs-vs-cats-100":
        from .dogs_vs_cats_100 import get_dogs_vs_cats_100 as get
    elif dataset == "gridfonts":
        from .gridfonts import get
    elif dataset == "figure-ground-a":
        from .gridfonts import get_figure_ground_a as get
    elif dataset in ["cmu-faces", "cmu-faces-full"]:
        from .cmu_faces import get_full as get
    elif dataset == "cmu-faces-half":
        from .cmu_faces import get_half as get
    elif dataset == "cmu-faces-quarter":
        from .cmu_faces import get_quarter as get
    elif dataset == "flowers":
        from .flowers import get
    elif dataset == "nanoGPT_shakespeare":
        from .nanoGPT_shakespeare import get_dataset as get
    else:
        raise Exception("unknown dataset name")
    return get()
