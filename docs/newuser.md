# New Users Guide

Welcome new user! This document will continue to be updated based on user feedback. This table quickly explains how to get things done in `PyEarthTools` 
|     Step       |    Without PyEarthTools     |    With PyEarthTools |
|----------------|-----------------------------|----------------------|
| Obtaining and loading data | Manual download or open from disk + data cleaning | Use or adapt in-built fetchers and openers (no cleaning) |
| Process, subset, augment, tranform and normalise data | Custom code | Use pre-defined validated Pipelines |
| Present your data to PyTorch or another framework as a Python iterator | Custom code to iterate over data | Pipelines are iterators |
| Define a machine learning model | Clone someone's repo or define your own | Use a 'bundled model' or write your own |
| Denormalise model outputs | Manual code | Pipelines are reversible |
| Model evaluation | Use a separate framework | Use pre-defined evaluation to generate standard scorecards |

This approach also allows your research to be more targetted into varying only the part of the end-to-end process which you are investigating. By starting with a baseline implementation to provide a strong basis for comparison and modifying only the relevant step, you can undertake a more controlled investigative process, confidently generating results from experimentation along the way.

## Installation

We strongly recommend using either a Conda or Python [virtual environment](installation.md#virtual-environments).

:::::{tab-set}
::::{tab-item} Conda environment
Run the following commands to install PyEarthTools in a Conda environment:
```shell
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
conda create -y -p ./venv python graphviz
conda activate ./venv
pip install -r requirements.txt
cd notebooks
jupyter lab
```
::::
::::{tab-item} Python virtual environment
Run the following commands to install PyEarthTools in a Python virtual environment:
```shell
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab
```
:::{admonition} Optional dependencies
:class: tip
Install [Graphviz](https://graphviz.org/download/) (not installable via pip) to display pipelines.
:::
::::
:::::

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
