# aitk: Artificial Intelligence Toolkit

[![DOI](https://zenodo.org/badge/339135763.svg)](https://zenodo.org/badge/latestdoi/339135763)

This collection contains two things: an open source set of Python tools, and a set of computational essays for exploring Artificial Intelligence, Machine Learning, and Robotics. This is a collaborative effort started by the authors, building on almost a century of collective experience in education and research.

The code and essays are designed to require as few computing resources as necessary, while still allowing readers to experience first-hand the topics covered.

## Authors

* [Douglas Blank](https://cs.brynmawr.edu/~dblank/) - Emeritus Professor of Computer Science, Bryn Mawr College; Head of Research at [Comet.ml](https://comet.ml/)
* [Jim Marshall](http://science.slc.edu/~jmarshall/) - Professor in the Computer Science Department at Sarah Lawrence College
* [Lisa Meeden](https://www.cs.swarthmore.edu/~meeden/) - Professor in the Computer Science Department at Swarthmore College

## Contributors

Please feel free to contribute to this collection: https://github.com/ArtificialIntelligenceToolkit/aitk

* Your Name Here

## Computational Essays

Each computational essay is described at [Computational Essays](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/ComputationalEssays.md).

## Python tools

`aitk` is a virtual Python package containing the following modules.

* [aitk]() - top level virtual package; install this to get all of the following
  * [aitk.robots](https://github.com/ArtificialIntelligenceToolkit/aitk.robots/) - Python package for exploring simulated mobile robots, with cameras and sensors
  * [aitk.algorithms](https://github.com/ArtificialIntelligenceToolkit/aitk.algorithms/) - Python package for exploring algorithms
  * [aitk.networks](https://github.com/ArtificialIntelligenceToolkit/aitk.networks/) - Python package for constructing and visualizing Keras deep learning models
  * [aitk.utils](https://github.com/ArtificialIntelligenceToolkit/aitk.utils/) - Python package for common utilities

### Python Installation

We recommend using `miniconda` for running Jupyter Notebooks locally on your computer. However, you can also skip this and run the Computational Essays on other services, such as Google's Colab. To use `miniconda`:

1. First install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Next, activate your base environment: `source ~/miniconda/bin/activate`
3. Create a Python 3.8 conda environment: `conda create --name py38 python=3.8` 
4. Activate it: `conda activate py38`

You only need to do step 1 once. To get out of conda, back to your regular system:

* `conda deactivate` (will get out of py38)
* `conda deactivate` (will get out of base environment)

### Software Installation

After activating your conda environment:

1. `pip install "aitk.robots[jupyter]"` (installs all of the aitk.robots requirements to run in Jupyter Lab 3.0)
2. `pip install pandas tensorflow numpy matplotlib tqdm ipycanvas` (some things you might want)

#### Jupyter Installation

If you want to work in notebooks and jupyter lab:

1. `pip install jupyterlab`  
2. `jupyter labextension install @jupyter-widgets/jupyterlab-manager ipycanvas`
3. `jupyter lab` starts it up, opens browser window

## AITK Community

For questions and comments, please use https://github.com/ArtificialIntelligenceToolkit/aitk/discussions/
