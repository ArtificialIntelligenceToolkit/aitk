# Computational Essays on Artificial Intelligence

This collection of essays is meant to be computationally executed. More details about this project can be found in the [README](./README.md). You can find all of these essays in the [notebooks folder.](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/notebooks)

Each of the notebooks contains both explanation and code. The notebooks are designed to be read and executed interactively to test out the concepts being explained.  For each subsection, we have a suggested ordering for exploring the available notebooks. Our notebooks and the sequencing can be found in the [notebooks folder](https://github.com/ArtificialIntelligenceToolkit/aitk/tree/master/notebooks) of this repo.

If you have never used a Jupyter notebook before, we recommend beginning with the [Intro To Jupyter Notebook](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/IntroToJupyterNotebook.ipynb) essay. You can run these online using Google CoLab (click the Open in CoLab button at the top of the essay) or download them to run in JupyterLab on your computer.

## Neural Networks 

#### [Basic Neural Nets](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/NeuralNetworks/BasicNeuralNets.ipynb)

Topics covered include an `introduction to neural networks` and `machine learning`.

> This essay explores a small dataset of handwritten digits. You will build and train a simple neural network to correctly classify the digits 0-9, and see its limitations when applied to noisy images.

#### [Categorizing Faces](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/NeuralNetworks/CategorizingFaces.ipynb)

Topics covered include `supervised learning` and `visualizing weights`. [Open in CoLab.]

> This essay explores the faces dataset created in conjunction with Tom Mitchell's 1977 book entitled *Machine Learning*. After training the network to recognize whether a person is wearing sunglasses or facing a particular direction, you will visualize the learned weights as an image to discover what aspects of the images the network has learned to focus on to perform its classification.

#### [Data Manipulation](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/NeuralNetworks/DataManipulation.ipynb)

Topics covered include `composition of data` and `biased outcomes`.

> This essay explores how the composition of the training data can lead to biased outcomes. It focuses on a subset of a larger dataset of handwritten digits from MNIST. By manipulating the numbers of 4's vs 5's in the dataset, it demonstrates that when the imbalance is large enough, the network struggles recognizing the underrepresented class. 

#### [Analyzing Hidden Representations](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/NeuralNetworks/AnalyzingHiddenRepresentations.ipynb)

Topics covered include `hidden space` and `Principal Components Analysis (PCA)`.

> This essay returns to the small dataset of handwritten digits covered in the notebook on Basic Neural Nets. Here, you will visualize the hidden layer representations prior to training and after training using Principal Components Analysis.


####  [Structure of Convolutional Neural Networks](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/NeuralNetworks/StructureOfConvolutionalNeuralNetworks.ipynb)

Topics covered include `Convolutional Neural Networks`.

> In this essay, we do a deep-dive into the functioning of a Convolutional Neural Network that is designed for processing two-dimensional image data.

## Robotics

#### [Braitenberg Vehicles](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/Robotics/BraitenbergVehicles.ipynb)

Topics covered include simple `Robot control`.

> This essay explores some of the vehicles described by Valentino Braitenberg in his book entitled *Vehicles: Experiments in Synthetic Psychology*. Although this essay doesn't explore Machine Learning, it does offer some insight into how an agent's behavior could be tightly coupled with its environment.

#### [What is it like to be a robot?](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/Robotics/WhatIsItLikeToBeARobot.ipynb)

Topics covered include `Philosophy of mind`, `Embodiment`, and `umwelt`.

> Philosopher Thomas Nagel famously asked "What is it like to be a bat?" In this essay, we explore the question "What is it like to be a robot?" Here, you explore first-hand what it is like to sense the world from a robot's perspective, and make your way through a simulated world.

#### [Demo Robots](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/Robotics/DemoRobots.ipynb)

Topics covered include how to create a `world`, `robot`, and `controller` in AITK

> This notebook demonstrates all of the key features of the AITK robotics tools.

#### [Subsumption](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/Robotics/Subsumption.ipynb)

Topics covered include Rodney Brooks' `subsumption architecture`

> In 1986, MIT Roboticist Rodney Brooks proposed a real-time, reactive framework for controlling robots called the subsumption architecture, which is based on a hierarchical collection of layers. This notebook demonstrates an example of this framework applied to the task of a robot trying to find a light source. 

#### [Seek Light](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/Robotics/SeekLight.ipynb)

Topics covered a case-based controller

> This notebook solves the same problem as the subsumption notebook, but does so using a case-based approach. The controller is tested on a series of more difficult worlds.

## Generative AI

#### [Word Embedding](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/GenerativeAI/WordEmbedding.ipynb)

Topics covered include how `word embeddings` are learned.

> This essay explores how a rich representation of a word's meaning can be inferred solely from its associations with other words. A stripped down language with just 27 words and a limited set of templates is examined. 

#### [NanoGPT](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/GenerativeAI/NanoGPT.ipynb)

Topics covered include `Large Language Models`

> This essay demonstrates a small-scale LLM applied to all of Shakespeare's plays.

#### [Transformer](https://github.com/ArtificialIntelligenceToolkit/aitk/blob/master/notebooks/GenerativeAI/Transformer.ipynb)

Topics covered include `transformers` and `attention` within LLMs.

> This essay goes into much more detail about how LLMs operate using the same NanoGPT model trained on the works of Shakespeare.
