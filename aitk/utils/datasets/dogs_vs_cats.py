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
    "dogs-vs-cats.tar.gz",
    "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/dogs-vs-cats/dogs-vs-cats.tar.gz",
    extract=True,
)

# Strip off gz, then tar:
DATA_DIR = os.path.splitext(os.path.splitext(_filename)[0])[0]

def make_npy(filename, match_filename):
    print("Creating %s..." % filename)
    # create npy file
    arrays = []
    for file in sorted(glob.glob(match_filename)):
        image = Image.open(file)
        array = np.array(image, dtype="float16") / 255.0
        arrays.append(array)
    np_array = np.array(arrays)
    np.save(filename, np_array)
    return np_array

def get_dogs():
    dogs_filename = os.path.join(DATA_DIR, "dogs.npy")
    if not os.path.exists(dogs_filename):
        dogs = make_npy(dogs_filename, os.path.join(DATA_DIR, "dog.*.jpg"))
    else:
        dogs = np.load(dogs_filename)

    target_dogs = [[1, 0] for i in range(len(dogs))]
    return dogs, np.array(target_dogs, dtype="float16")

def get_cats():
    cats_filename = os.path.join(DATA_DIR, "cats.npy")
    if not os.path.exists(cats_filename):
        cats = make_npy(cats_filename, os.path.join(DATA_DIR, "cat.*.jpg"))
    else:
        cats = np.load(cats_filename)

    target_cats = [[0, 1] for i in range(len(cats))]
    return cats, np.array(target_cats, dtype="float16")

def get():
    """
    """
    dogs, target_dogs = get_dogs()
    cats, target_cats = get_cats()

    return (np.concatenate((dogs, cats)),
            np.concatenate((target_dogs, target_cats)))
