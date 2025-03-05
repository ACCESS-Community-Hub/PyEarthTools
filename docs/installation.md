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

The supported installation options are:

- user: The default option, containing the functionality. 
- dev: Also contains developer tools 

## Repository Layout

This is a so-called monorepo. PyEarthTools comprises multiple, modular packages within a shared namespace that inter-operate in order to provide the overall functionality of the framework. It is not necessary to install all of them, and it is envisioned that many users are likely to want only some parts of the framework. As such, each sub-package is a fully independent Python package, with its own requirements and its own installation process. Each of these sub-packages lies in the `packages` subdirectory.

### User installation

Each of PyEarthTools package can be installed separately using `pip`, directly from GitHub.
For example, to install the `utils` sub-package, use:

```
pip install "pyearthtools[utils] @ git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git"
```

Other available packages are `data`, `pipeline` and `training`.

To install all PyEarthTools packages, including all their optional dependencies, use:

```
pip install "pyearthtools[all] @ git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git"
```

## Developer installation

Developers of PyEarthTools will most likely want to check out the entire monorepo and work on changesets which may span sub-packages. Each sub-package is versioned separately, so bugfixes or updates in a single sub-package can be performed independently without requiring a new release of the entire ecosystem. 

First clone this repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools
```

and install all packages in "editable" mode with

```
pip install -r requirements-dev.txt
```

or install a specific package `<package-name>` in editable mode using

```
pip install -e packages/<package-name>
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


> [!WARNING]
> These instructions have been tested on Linux and macOS. We have not tested them on **Windows**.
> We welcome any contribution to improve this situation 🙂.

