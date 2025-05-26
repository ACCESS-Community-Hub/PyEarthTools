# Installation Guide

This page describes different ways to install PyEathTools depending on intended usage:

- run tutorials content (recommended for new users),
- install PyEarthTools packages as dependencies in your Python project,
- install PyEarthTools in developer mode in order to contribute.

These installation instructions have been tested on Linux, maxOS and Windows. If you encounter any difficulties, please [raise an issue](https://github.com/ACCESS-Community-Hub/PyEarthTools/issues).

## Virtual Environments

Users installing PyEarthTools for themselves (such as on their own workstation or laptop) are recommended to use a virtual environment. The instructions below proceed on this assumption.

Users in shared computing environments (as is common in HPC and other research facilities) may or may not be able to use virtual environments easily, and may or may not be able to utilise `conda`. The instructions below are based on initially creating a virtual environment with `conda` and then installing the packages using `pip`. Notes are included in this section about the additional steps required if using `pipenv` or `venv` to create the virtual environment.

Virtual environments are isolated, dedicated copies of Python, which are separate from the version of Python which may be present and used by other software or your operating system. Using a virtual environment avoids the need for `root` or `Administrator` access, and also lowers the risk of corrupting the system if there are any problems with the installation. Industry standard practice for software development is to use virtual environments in this way.

For those wishing to use `pipenv` or `venv` instead of `conda`, you will need to install two additional packages into your system which are not supplied by `pip`. These are `graphviz` and `pandoc`. While there is a package called `graphviz` in pip, it only supplies Python wrappers around the core package which must be installed separately. The `conda` framework does allow the full installation of `graphviz`. Neither of these two packages are completely essential. `graphviz` is used for the display of pipelines, but is not core to the functionality and its absence will not result in unhandled exceptions. `pandoc` is only required for building the documentation locally.

## Creating a Virtual Environment 

Here is a command to create and activate a new virtual environment with *conda*:
```py
conda create --name <my-env> python
conda activate <my-env>
```

Here is a command to create and activate a new virtual environment with *conda*, into a specified directory (often required when on shared computing facilities):
```py
conda create -p <path_to_environment> python
conda activate -p <path_to_environment>
```

Here is a command to create and activate a new virtual environment with *venv*:
```py
python -m venv <path_to_environment>
source <path_to_environment>/bin/activate
```

These approaches will all create a new virtual environment, with the Python interpreter installed, but no additional packages yet.

## Using or Developing PyEarthTools at NCI

A very large number of the users of PyEarthTools work on the NCI supercomputing environment. Users here should request access to project code `dk92`. The modules environment `/g/data/dk92/apps/Modules/modulefiles` can then be specified, and the module `pet/2025.05` can be used (noting this module name will change with each new update). This will make the most recent update of PyEarthTools available. 

For developers of PyEarthTools at NCI, this is the recommended approach, nt a virtual environment. Users can then use `pip install -e` to check out the latest code from the repository, and Python will install the development packages into a userspace install directory.


## Installation of PyEarthTools (including tutorials)

This section details how to install PyEarthTools to be able run notebooks from the [](notebooks/Gallery.ipynb). It assumed that users want to run PyEarthTools but don't wish to change its core code.

First, make sure to have [Git](https://git-scm.com/) and [Conda](https://conda-forge.org/download/) installed on your system.

Then, clone the PyEarthTools repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools
```

and create a Conda environment to install tutorials dependencies:

```
conda env create -f tutorials.yml -p ./venv
```

You can start a JupyterLab instance to run the example notebooks:

```
conda run -p ./venv --no-capture-output jupyter-lab notebooks/
```

````{Note}
Alternatively, you can install a Jupyter kernel to run notebooks in a pre-existing JupyterLab installation:

```
conda run -p ./venv --no-capture-output \
    python -m ipykernel install --user --name PET-tutorial
```

See the [IPython documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) for additional information regarding the IPython kernel installation.
````

## Installing Individual Packages

PyEarthTools comprises multiple, modular packages within a shared namespace that inter-operate in order to provide the overall functionality of the framework.

It is not necessary to install all of them, and it is envisioned that many users are likely to want only some parts of the framework.

Each PyEarthTools package can be installed separately using `pip`, directly from GitHub.
For example, to install the `utils` sub-package, use:

```
pip install "pyearthtools[utils] @ git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git"
```

Other available packages are `data`, `pipeline`, `training`, `tutorial` and `zoo`.

To install all PyEarthTools packages, including all their optional dependencies, use:

```
pip install "pyearthtools[all] @ git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git"
```

## Developer Installation

PyEarthTools code is organised as a monorepo, each sub-package lies in a different sub-directory in the `packages` directory.

Developers of PyEarthTools will most likely want to check out the entire monorepo and work on changesets which may span sub-packages. 

The following instructions detail how to install PyEarthTools in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), making it easier to implement and test changes iteratively.

```{tip}
Each sub-package is versioned separately, so bugfixes or updates in a single sub-package can be performed independently without requiring a new release of the entire ecosystem.
```

First clone the PyEarthTools repository:

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

