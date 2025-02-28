# Detailed Installation Guide

## Overview

This page describes:

- Setting up a virtual environment.
- The most common installation options for PyEarthTools. (Expert users of pip and conda will note that more variations are possible.)
- An advanced installation option for Jupyter Notebook, for users who wish to separate the Jupyter environment and the PyEarthTools execution environment.

## Setting up a Virtual Environment

In almost all cases, it is recommended to use a virtualised Python environment. 

PyEarthTools can be installed using either venv/pip or conda/pip. 

Here is a command to create and activate a new virtual environment with *virtualenv*:

```py
python -m venv <path_to_environment>
source <path_to_environment>/bin/activate
```

Here is a command to create and activate a new virtual environment with *conda*:
```py
conda create --name <my-env>
conda activate <my-env>
```

## Installation Options

There are multiple installation options. Most users currently want the "all" installation option. 

The supported installation options are:

- user: The default option, containing the functionality. 
- dev: Also contains developer tools 

### User installation

Each of PyEarthTools package can be installed separately using `pip`, directly from GitHub.
For example, to install the `pyearthtools-utils` package, use:

```
pip install git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git#subdirectory=packages/utils
```

Other available packages are `pyearthtools-data`, `pyearthtools-pipeline` and `pyearthtools-training`, that can be installed as follows:

```
pip install git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git#subdirectory=packages/data
pip install git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git#subdirectory=packages/pipeline
pip install git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git#subdirectory=packages/training
```

## Developer installation

Developers of `PyEarthTools` will most likely want to check out the entire monorepo and work on changesets which may span sub-packages. Each sub-package is versioned separately, so bugfixes or updates in a single sub-package can be performed independently without requiring a new release of the entire ecosystem. 

First clone this repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
```

and install all packages in "editable" mode with

```
cd PyEarthTools/packages/<package-name>
pip install -e ".[dev]"
```

### Jupyter Notebook - Advanced Installation Option

Some users may wish to separate the Jupyter environment and the PyEarthTools execution environment. One way to achieve this is by creating a new PyEarthTools virtual environment and registering it as a new kernel within another Jupyter environment. You can then run the tutorials and/or execute PyEarthTools code within the kernel. Registering the kernel can be done as follows:

1. Determine the "prefix" of the Jupyter environment. 
2. Choose a name to use for a new kernel.
3. Activate the PyEarthTools virtual environment which will be used as the kernel.
4. Execute the registration command.

A sample command to register a new kernel is:

`python -m ipykernel install --user --prefix=<path-to-server-environment> --name=<pick-any-name-here>`

[https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html](https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html) provides additional technical details regarding the registration of kernels.

