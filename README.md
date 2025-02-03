# PyEarthTools: Reproducible science pipelines for machine learning

`PyEarthTools` is a Python framework, containing modules for loading data; pre-processing, normalising and standardising data; defining machine learning (ML) models; training ML models; performing inference with ML models; and evaluating ML models. It contains specialised support for weather and climate data sources and models. It has an emphasis on reproducibility, shareable pipelines, and human-readable low-code pipeline definition.

> [!NOTE]
> * THIS REPOSITORY IS UNDER CONSTRUCTION *
>
> This repository contains code which is under construction, and should not yet be used by anyone.
> The development team are actively working to make this project ready for new users, but for
> the time being things are not ready. Feel free to take a look around if you like, but much is likely
> to change in the next few months.
>

**Installation**

First clone the repository using the `monorepo` branch:

```
git clone git@github.com:informatics-lab/pyearthtools.git
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

To run example [notebooks from the ERA5 low-res package](packages/era5lowres/nbook), you may need to also install a Jupyter kernel for your environment:

```
# from the activate virtual or conda environment
pip install ipykernel
python -m ipykernel install --user --name pyearthtools
```
