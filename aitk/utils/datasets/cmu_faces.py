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

from .utils import get_file, Dataset


def load_data(data_file):
    np_data = np.load(data_file, allow_pickle=True)

    labels = np_data["labels"]

    return Dataset(
        train_inputs=np_data["data"].tolist(),
        train_features=[labels.split("_") for labels in labels.tolist()]
    )

def get_full():
    """
    624, 120, 128
    """
    data_file = get_file(
        "cmu_faces_full_size.npz",
        "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/cmu_faces/cmu_faces_full_size.npz",
        extract=True,
    )
    return load_data(data_file)

def get_half():
    """
    """
    data_file = get_file(
        "cmu_faces_half_size.npz",
        "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/cmu_faces/cmu_faces_half_size.npz",
        extract=True,
    )
    return load_data(data_file)

def get_quarter():
    """
    """
    data_file = get_file(
        "cmu_faces_quarter_size.npz",
        "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/cmu_faces/cmu_faces_quarter_size.npz",
        extract=True,
    )
    return load_data(data_file)

