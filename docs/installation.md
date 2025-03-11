# Detailed Installation Guide

## Overview

This page describes different ways to install PyEathTools depending on intended usage:

- run tutorials content (recommended for new users),
- install PyEarthTools packages as dependencies in your Python project,
- install PyEarthTools in developer mode in order to contribute.

## Tutorials installation

This section details how to install PyEarthTools to be able run notebooks from the tutorial gallery.

First, make sure to have [Git](https://git-scm.com/) and [Conda](https://conda-forge.org/download/) installed on your system.

:::{warning}
These instructions have been tested on Linux and macOS. We have not tested them on **Windows**.
We welcome any contribution to improve this situation 🙂.
:::

Then, clone the PyEarthTools repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
```

and create a Conda environment to install tutorials dependencies:

```
conda env create -f tutorials.yml -p ./venv
```

Finally, to run the example notebooks, you can either

- start a JupyterLab instance

```
conda activate ./venv
jupyter lab
```

- or install a Jupyter kernel to use in a pre-existing JupyterLab installation

```
conda activate ./venv
python -m ipykernel install --user --name PET-tutorial
```

Tutorial notebooks are accessible under the `tutorials` folder.

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

