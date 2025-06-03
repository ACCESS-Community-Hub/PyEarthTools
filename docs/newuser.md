# New Users Guide

Welcome new user! This document will continue to be updated based on user feedback. 

## Installation

From within a suitable [virtual environment](installation.md#virtual-environments), run the following commands:

```
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
pip install -r requirements-dev.txt
conda install graphviz
cd notebooks
jupyter lab
```

For other installation options, please refer to the [installation guide](installation.md).

## Where to Start

The tutorial ["Train and run a simplified global weather model"](./notebooks/tutorial/FourCastMini_Demo.ipynb) is the best place to start if you are working in your own environment. This tutorial has been tested with a 4GB GPU, uses less than 3GB of training data, and each model training epoch will take between 10 and 25 minutes depending on your hardware. This tutorial will also work at NCI or on other HPC facilities.

If you are working at NCI, then ["Blending Data from Multiple Sources"](./notebooks/tutorial/MultipleSources.ipynb) and ["Working with Climate Data"](./notebooks/tutorial/Working_with_Climate_Data.ipynb) are also good places to start. These tutorials both use very large data sets. These data sets are archived on disk at NCI so these tutorials are straightforward to run using NCI facilities.

## Core Concepts in PyEarthTools

A modelling project in PyEarthTools involves the following steps:

1. Fetching and loading data
2. Processing data for machine learning 
3. Training a model
4. Evaluating the model

The ["Train and run a simplified global weather model"](./notebooks/tutorial/FourCastMini_Demo.ipynb) tutorial demonstrates the first three of these steps. Guidance for new users on model evaluation will be added at a later date.



