# DeepJDOT

This repository contains the codes of the following paper

BB Damodaran, B Kellenberger, R Flamary, D Tuia, N Courty, "[DeepJDOT:Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation](https://arxiv.org/abs/1803.10081)", in European Conference on Computer Vision 2018 (ECCV-2018).

## Dependencies

In order to run, the code requires the following Python modules:

* Numpy
* Matplotlib
* [POT](https://github.com/rflamary/POT) (Python Optimal Transport library)
* keras with tensorflow backend (preferably GPU version)
* imutils (only for rotating images in regression demo)
* scikit-learn (for scoring functions)

## Modules

* Deepjdot - module contains the implementation of the DeepJDOT
* dnn      - import necessary functions from keras
* deepjdot_demo  - DeepJDOT on the sample dataset
* deepjdot_svhn_mnist - DeepJDOT on SVHN & MNIST dataset

To run the DeepJDOT on the sample 2D dataset, please see or run the "deepjdot_demo.py".

To run on the real data set: SVHN --> MNIST, please see "deepjdot_svhn_mnist.py".
The default task is to do classification but you can turn on regression demo by setting `do_reg` to `True`.
If you do not want to wait long training time you can set `small_model` to `True`.

For regression demo, each image will be randomly rotated around its center, and then the label will be the angle rotated.
The angle will be scaled to [0, 1]. The model needs to predict how much the image is rotated.

I suggest you run the demo files inside `Spyder` or any interactive python IDE so that you can investigate
each cell denoted by `#%%` lines and understand the code better.
