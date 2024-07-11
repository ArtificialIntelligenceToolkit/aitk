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
import matplotlib.pyplot as plt

def get_images(flower):
    if flower == "flower1":
        url = "https://github.com/ArtificialIntelligenceToolkit/datasets/blob/c3b329d9519ebacab99b20d0c3c78b21a5c2a6c2/noise_and_embeddings/flower1.jpg"
        os.system(f'wegt {url} -0 flower1.jpg')
        input = plt.imread('flower1.jpg')
    elif flower == "flower2":
        url = "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/noise_and_embeddings/flower.webp"
        os.system(f'wegt {url} -0 flower.webp')
        input = plt.imread('flower.webp')
    else:
        url = "https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/noise_and_embeddings/flower3.webp"
        os.system(f'wegt {url} -0 flower3.webp')
        input = plt.imread('flower3.webp')

    return input

