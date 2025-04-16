# Installation Guide

This page describes different ways to install PyEathTools depending on intended usage:

- run tutorials content (recommended for new users),
- install PyEarthTools packages as dependencies in your Python project,
- install PyEarthTools in developer mode in order to contribute.

These installation instructions have been tested on Linux, maxOS and Windows. If you encounter any difficulties, please [raise an issue](https://github.com/ACCESS-Community-Hub/PyEarthTools/issues).

## Tutorials Installation

This section details how to install PyEarthTools to be able run notebooks from the [](notebooks/Gallery.ipynb).

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

## Package Installation

PyEarthTools comprises multiple, modular packages within a shared namespace that inter-operate in order to provide the overall functionality of the framework.
It is not necessary to install all of them, and it is envisioned that many users are likely to want only some parts of the framework.

Each PyEarthTools package can be installed separately using `pip`, directly from GitHub.
For example, to install the `utils` sub-package, use:

```
pip install "pyearthtools[utils] @ git+https://github.com/ACCESS-Community-Hub/PyEarthTools.git"
```

Other available packages are `data`, `pipeline` and `training`.

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

```{note}
For notebooks development, use the [tutorials installation](#tutorials-installation) instructions instead.
```
