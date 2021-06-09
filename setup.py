# -*- coding: utf-8 -*-
# *******************************************************
# aitk: Python tools for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk/
#
# *******************************************************

"""
aitk setup
"""
import io
import os

import setuptools

name = "aitk"

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md"), "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name=name,
    version="1.0.9",
    url="https://github.com/ArtificialIntelligenceToolkit/%s" % name,
    author="Douglas Blank",
    description="Python tools for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "aitk.algorithms>=0.0.5",
        "aitk.robots>=0.9.26",
        "aitk.networks>=0.3.11",
        "aitk.utils>=0.4.3"
    ],
    python_requires=">=3.6",
    license="BSD-3-Clause",
    platforms="Linux, Mac OS X, Windows",
    keywords=["ai", "artificial intelligence", "robots",
              "simulator", "jupyter", "python", "machine learning",
              "neural networks", "keras", "tensorflow"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
