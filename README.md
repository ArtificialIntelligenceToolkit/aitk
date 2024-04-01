# aitk: Artificial Intelligence Toolkit

[![DOI](https://zenodo.org/badge/339135763.svg)](https://zenodo.org/badge/latestdoi/339135763)

This collection contains two things: an open source set of Python tools, and a set of computational essays for exploring Artificial Intelligence, Machine Learning, and Robotics. This is a collaborative effort started by the authors, building on almost a century of collective experience in education and research.

The code and essays are designed to require as few computing resources as necessary, while still allowing readers to experience first-hand the topics covered.

## Authors

* [Douglas Blank](https://github.com/dsblank/) - Emeritus Professor of Computer Science, Bryn Mawr College; Head of Research at [Comet.ml](https://comet.ml/)
* [Jim Marshall](http://science.slc.edu/~jmarshall/) - Professor in the Computer Science Department at Sarah Lawrence College
* [Lisa Meeden](https://www.cs.swarthmore.edu/~meeden/) - Professor in the Computer Science Department at Swarthmore College

## Contributors

Please feel free to contribute to this collection: https://github.com/ArtificialIntelligenceToolkit/aitk

* Your Name Here

## Computational Essays

Each computational essay is described at [Computational Essays](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/ComputationalEssays.md).

## Artifical Intelligence Toolkit

`aitk` is Python package containing the following modules.

* [aitk]() - top level package
  * [aitk.robots](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/docs/robots) - for exploring simulated mobile robots, with cameras and sensors
  * [aitk.algorithms](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/docs/algorithms/) - for exploring algorithms
  * [aitk.networks](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/docs/networks/) - for constructing and visualizing Keras deep learning models
  * [aitk.utils](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/docs/utils/) - for common utilities

### Python Installation

#### Using pip

If you already have an environment for running Python, and optionally
Jupyter Notebooks, you can simply execute this at the command line:

```
pip install aitk
```

If you haven't install Jupyter (and are not running in Google's
colab), jump down to "Jupyter Installation".

If you are inside a notebook (say on Google's colab):

```
%pip install aitk --quiet
```

#### Using conda

If you are setting up your own Jupyter Notebook environment on your
own computer, we recommend using `miniconda`.

To use `miniconda`:

1. First install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Next, activate your base environment: `source ~/miniconda/bin/activate`
3. Create a Python 3.8 conda environment: `conda create --name py38 python=3.8`
4. Activate it: `conda activate py38`

You only need to do step 1 once. To get out of conda, back to your regular system:

* `conda deactivate` (will get out of py38)
* `conda deactivate` (will get out of base environment)

### Software Installation

After activating your conda environment:

1. `pip install "aitk[jupyter]"` (installs all of the requirements to run in Jupyter Lab 3.0)
2. `pip install pandas tensorflow numpy matplotlib tqdm ipycanvas` (some things you might want)

#### Jupyter Installation

If you want to work in notebooks and jupyter lab:

1. `pip install jupyterlab`
2. `jupyter labextension install @jupyter-widgets/jupyterlab-manager ipycanvas`
3. `jupyter lab` starts it up, opens browser window

## AITK Community

For questions and comments, please use https://github.com/ArtificialIntelligenceToolkit/aitk/discussions/
