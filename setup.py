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
import json

import setuptools

name = "aitk"
HERE = os.path.abspath(os.path.dirname(__file__))

# Get our version
def get_version(file, name="__version__"):
    """Get the version of the package from the given file by
    executing it and extracting the given `name`.
    """
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]

with open(os.path.join(HERE, "README.md"), "r") as fh:
    long_description = fh.read()

version = get_version(os.path.join(HERE, "aitk/_version.py"))

setup_args = dict(
    name=name,
    version=version,
    url="https://github.com/ArtificialIntelligenceToolkit/%s" % name,
    author="Douglas Blank",
    description="Python tools for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "aitk.utils": ["fonts/*.ttf"],
        "aitk.robots": ["worlds/*.json", "worlds/*.png"],
    },
    install_requires=["Pillow", "ipywidgets", "tqdm", "numpy", "matplotlib", "tensorflow>=2.17.0"],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    license="BSD-3-Clause",
    platforms="Linux, Mac OS X, Windows",
    keywords=["ai", "artificial intelligence", "robots",
              "simulator", "jupyter", "python", "machine learning",
              "neural networks", "keras", "tensorflow"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
