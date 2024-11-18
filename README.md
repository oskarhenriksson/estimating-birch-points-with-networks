# Maximum likelihood estimation of log-affine models using detailed-balanced reaction networks
This repository contains code for the paper [Maximum likelihood estimation of log-affine models using detailed-balanced reaction networks](https://arxiv.org/abs/2411.07986) by Oskar Henriksson, Carlos Améndola, Jose Israel Rodriguez, and Polly Y. Yu.

## Contents
General code for estimating eigenvalues and timescales at the positive steady state for the network $G_{\Lambda,c}$ introduced in the paper can be found in the files `MLE_estimator_IPS.py` and 
`MLE_estimator_tellurium.py`. Code for generating the figures found §5.3 of the paper can be found in the Jupyter notebook `experiments.ipynb`.

## Dependencies
The code is based on Python 3.10.12, using the packages NumPy 1.26.4, Scipy 1.11.4, and Tellurium 2.2.10.
