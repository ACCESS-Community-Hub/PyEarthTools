# PyEarthTools: Reproducible science pipelines for machine learning

`PyEarthTools` is a Python framework, containing modules for loading data; pre-processing, normalising and standardising data; defining machine learning (ML) models; training ML models; performing inference with ML models; and evaluating ML models. It contains specialised support for weather and climate data sources and models. It has an emphasis on reproducibility, shareable pipelines, and human-readable low-code pipeline definition.

> [!NOTE]
> **THIS REPOSITORY IS UNDER CONSTRUCTION**
>
> This repository contains code which is under construction, and should not yet be used by anyone.
> The development team are actively working to make this project ready for new users, but for
> the time being things are not ready. Feel free to take a look around if you like, but much is likely
> to change in the next few months.
>

# Getting Started

Guidelines for new users still need to be developed.
For new users, we recommend following the instructions in the [Tutorials](#tutorials) section of this README.
Installation instructions are intended for users already familiar with the project.

# Installation

## Repository Layout

This is a so-called monorepo. `PyEarthTools` comprises multiple, modular packages within a shared namespace that inter-operate in order to provide the overall functionality of the framework. It is not necessary to install all of them, and it is envisioned that many users are likely to want only some parts of the framework. As such, each sub-package is a fully independent Python package, with its own requirements and its own installation process. Each of these sub-packages lies in the [`packages`](packages/) subdirectory.

## User installation

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

> [!NOTE]
> When this repository is ready for wider use, the intention is to release `PyEarthTools` on PyPI and conda-forge.

## Developer installation

Developers of `PyEarthTools` will most likely want to check out the entire monorepo and work on changesets which may span sub-packages. Each sub-package is versioned separately, so bugfixes or updates in a single sub-package can be performed independently without requiring a new release of the entire ecosystem. 

First clone this repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
```

and install all packages in "editable" mode with

```
cd PyEarthTools
pip install -r requirements-dev.txt
```

> [!WARNING]
> We do recommend using a Python virtual environment or a Conda environment when developing, to isolate this installation from the rest of your system.
> Examples are provided in the [Tutorials](#tutorials) section.


# Tutorials

First clone the repository:

```
git clone https://github.com/ACCESS-Community-Hub/PyEarthTools.git
cd pyearthtools
```

Then create a Python virtual environment:

```
python3 -m venv venv
venv/bin/activate
```

or a Conda environment to install all dependencies:

```
conda create -p ./venv -y python=3.11
conda activate ./venv
```

And install all dependencies via pip:

```
pip install -r requirements-dev.txt
```

You can run some of the tests to check that your installation worked:

```
pytest packages/data/tests/
```

To run example [notebooks from the tutorial package](packages/tutorial/nbook), you may need to also install a Jupyter kernel for your environment:

```
# from the activate virtual or conda environment
pip install ipykernel
python -m ipykernel install --user --name pyearthtools
```
