# Computational Essays on Artificial Intelligence

This collection of essays is meant to be computationally executed. More details about this project can be found in the [README](./README.md).

Each of the notebooks contains both explanation and code. The notebooks are designed to be read and executed interactively to test out the concepts being explained. Within each sub-section we have a suggested ordering for exploring the available notebooks.

If you have never used a Jupyter notebook before, we recommend beginning with the [IntroToJupyterNotebook.ipynb](https://colab.research.google.com/github/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/IntroToJupyterNotebook.ipynb) notebook.

## Neural Networks 

#### Basic Neural Nets

Topics covered include an `introduction to neural networks` and `machine learning`.

> This essay explores a small dataset of handwritten digits. You will build and train a simple neural network to correctly classify the digits 0-9, and see its limitations when applied to noisy images.

#### Categorizing Faces

Topics covered include `supervised learning` and `visualizing weights`.

> This essay explores the faces dataset created in conjunction with Tom Mitchell's 1977 book entitled *Machine Learning*. After training the network to recognize whether a person is wearing sunglasses or facing a particular direction, you will visualize the learned weights as an image to discover what aspects of the images the network has learned to focus on to perform its classification.

#### Data Manipulation

Topics covered include `composition of data` and `biased outcomes`.

> This essay explores how the composition of the training data can lead to biased outcomes. It focuses on a subset of a larger dataset of handwritten digits from MNIST. By manipulating the numbers of 4's vs 5's in the dataset, it demonstrates that when the imbalance is large enough, the network struggles recognizing the underrepresented class. 

#### Analyzing Hidden Representations

Topics covered include `hidden space` and `Principal Components Analysis (PCA)`.

> This essay returns to the small dataset of handwritten digits covered in the notebook on Basic Neural Nets. Here, you will visualize the hidden layer representations prior to training and after training using Principal Components Analysis.


####  Structure of Convolutional Neural Networks

Topics covered include `Convolutional Neural Networks`.

> In this essay, we do a deep-dive into the functioning of a Convolutional Neural Network that is designed for processing two-dimensional image data.

## Robotics

#### Braitenberg Vehicles

Topics covered include simple `Robot control`.

> This essay explores some of the vehicles described by Valentino Braitenberg in his book entitled *Vehicles: Experiments in Synthetic Psychology*. Although this essay doesn't explore Machine Learning, it does offer some insight into how an agent's behavior could be tightly coupled with its environment.

#### What is it like to be a robot?

Topics covered include `Philosophy of mind`, `Embodiment`, and `umwelt`.

> Philosopher Thomas Nagel famously asked "What is it like to be a bat?" In this essay, we explore the question "What is it like to be a robot?" Here, you explore first-hand what it is like to sense the world from a robot's perspective, and make your way through a simulated world.

#### Demo Robots

Topics covered include how to create a `world`, `robot`, and `controller` in AITK

> This notebook demonstrates all of the key features of the AITK robotics tools.

#### Subsumption

Topics covered include Rodney Brooks' `subsumption architecture`

> In 1986, MIT Roboticist Rodney Brooks proposed a real-time, reactive framework for contolling robots called the subsumption architecture, which is based on a hierarchical collection of layers. This notebook demonstrates an example of this framework applied to the task of a robot trying to find a light source. 

#### Seek Light

Topics covered a case-based controller

> This notebook solves the same problem as the subsumption notebook, but does so using a case-based approach. The controller is tested on a series of more difficult worlds.

## Components of Large Language Models

#### Word Embedding

Topics covered include how `word embeddings` are learned.

> This essay explores how a rich representation of a word's meaning can be inferred soley from its associations with other words. A stripped down language with just 27 words and a limited set of templates is examined. 

#### NanoGPT

Topics covered include `Large Language Models`

> This essay demonstrates a small-scale LLM applied to all of Shakespear's plays.
