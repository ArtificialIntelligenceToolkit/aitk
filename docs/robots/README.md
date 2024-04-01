# aitk.robots

[![PyPI version](https://badge.fury.io/py/aitk.robots.svg)](https://badge.fury.io/py/aitk.robots) [![CI build status](https://img.shields.io/circleci/build/github/ArtificialIntelligenceToolkit/aitk.robots)](https://app.circleci.com/pipelines/github/ArtificialIntelligenceToolkit/aitk.robots)

A lightweight Python robot simulator for JupyterLab, Notebooks,
and other Python environments.

<img src="https://artificialintelligencetoolkit.github.io/aitk.robots/images/hello-world.png"></img>

## Goals

1. A lightweight mobile robotics simulator
2. Usable in the classroom, research, or exploration
3. Explore wheeled robots with range, cameras, smell, and light sensors
4. Operate quickly without a huge amount of resources
5. Create reproducible experiments
6. Designed for exposition, experimentation, and analysis
7. Sensors designed for somewhat realistic problems (such as image recognition)
8. Especially designed to work easily with Machine Learning and Artificial Intelligence systems

## Installation

For the core operations, you will need to install just aitk.robots:

```shell
pip install aitk.robots
```

To use the Jupyter enhancements, you'll also need the browser-based
extensions. You can install those with:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

If not in a conda environment, then you will also need to:

```
jupyter nbextension enable --py widgetsnbextension
```

For additional information, please see:

* [aitk](https://github.com/ArtificialIntelligenceToolkit/aitk)
* [aitk.robots](https://github.com/ArtificialIntelligenceToolkit/aitk.robots)
* [aitk.networks](https://github.com/ArtificialIntelligenceToolkit/aitk.networks)
* [aitk.utils](https://github.com/ArtificialIntelligenceToolkit/aitk.utils)
