# -*- coding: utf-8 -*-
# **************************************************
# aitk: Python tools for AI
#
# Copyright (c) 2021 ArtificialIntelligenceToolkit
#
# https://github.com/ArtificialIntelligenceToolkit/aitk/
#
# **************************************************

"""
aitk setup
"""
import io
import os

import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

# The name of the project
name = "aitk"


with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name=name,
    version="1.0.0",
    url="https://github.com/ArtificialIntelligenceToolkit/%s" % name,
    author="Douglas Blank",
    description="Python tools for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["aitk.robots", "aitk.networks", "aitk.utils"],
    python_requires=">=3.6",
    license="BSD-3-Clause",
    platforms="Linux, Mac OS X, Windows",
    keywords=["robot", "simulator", "jupyter", "python"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
