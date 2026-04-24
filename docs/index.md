# PyEarthTools: Machine learning for Earth system science

[![DOI](https://zenodo.org/badge/903938118.svg)](https://doi.org/10.5281/zenodo.15760768)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FACCESS-Community-Hub%2FPyEarthTools%2Frefs%2Fheads%2Fdevelop%2Fpyproject.toml&label=Python)

- A Python framework that supports the develoment of machine learning models, big and small, for Earth system science
- Suitable for students and newcomers, as well as for domain specialists and scientists
- Runs effectively on HPC (supercomputers), cloud, workstations and laptops
- Suitable for use with megabytes to petabytes of data
- Still under early-stage development - things are likely to change a lot. If you notice an issue, please feel free to raise it on GitHub

<figure style="display:inline-block; width:45%; margin-right:5%;">
    <img src="https://pyearthtools.readthedocs.io/en/latest/_images/notebooks_demo_FourCastNeXt_Inference_9_1.png" alt="A prediction of the weather" width="100%">
    <figcaption>A weather prediction from a model trained with PyEarthTools.</figcaption>
</figure>

<figure style="display:inline-block; width:45%; vertical-align:top;">
    <img src="https://pyearthtools.readthedocs.io/en/latest/_images/notebooks_tutorial_Working_with_Climate_Data_14_2.svg" alt="A data processing pipeline" width="300">
    <figcaption>A data processing flow composed for working with climate data.</figcaption>
</figure>

Source Code: [github.com/ACCESS-Community-Hub/PyEarthTools](https://github.com/ACCESS-Community-Hub/PyEarthTools)
Documentation: [pyearthtools.readthedocs.io](https://pyearthtools.readthedocs.io)
Tutorial Gallery: [available here](./notebooks/Gallery)
New Users Guide: [available here](newuser.md)

**If you use `PyEarthTools` for your work or a publication, [please cite our work](https://pyearthtools.readthedocs.io/en/latest/#citing-pyearthtools).**

## Installation

**Here is the quickest way to install the complete framework and get started:**

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

PyEarthTools comprises multiple sub-packages which may be installed and used separately. See the [installation guide](installation.md) for more details.

## Overview of PyEarthTools

PyEarthTools is a Python framework containing modules for:
 - loading and fetching data;
 - pre-processing, normalising and standardising data into a normal form suitable for machine learning;
 - defining machine learning (ML) models;
 - training ML models and managing experiments;
 - performing inference with ML models;
 - and evaluating ML models (coming soon).

PyEarthTools runs effectively on HPC (supercomputers), cloud, workstations and laptops.

## Overview of the Packages within PyEarthTools

PyEarthTools comprises multiple sub-packages which can be used individually or together.

|    Sub-Package                                         |  Purpose  |
|--------------------------------------------------------|---------------------- |
|  [Data](api/data/data_index.md)                        | Loading and indexing Earth system data into xarray |
|  [Utils](api/utils/utils_index.md)                     | Code for common functionality across the sub-packages |
|  [Pipeline](api/pipeline/pipeline_index.md)            | Process and normalise Earth system data ready for machine learning |
|  [Training](api/training/training_index)               | Training processes for machine learning modelsl |
|  [Tutorial](api/tutorial/tutorial_index.md)            | Contains helper code for data sets used in tutorials |
|  [Bundled Models](api/bundled_models/bundled_index.md) | Maintained versions of specific, bundled models which can be easily trained and run |
|  [Zoo](api/zoo/zoo_index.md)                           | Contains code for managing registered models (such as the bundled models) |
|  Evaluation                                            | (Coming soon) Contains code for producing standard evaluations (such as benchmarks and scorecards) |

## Citing `PyEarthTools`

If you use PyEarthTools for your work, we would appreciate you citing our software as below:

:::::{tab-set}
::::{tab-item} APA
Leeuwenburg, T., Cook, H., Rio, M., Hobeichi, S., Miller, J., Mason, G., Ramanathan, N., Pill, J., Haddad, S., Stassen, C., de Burgh-Day, C., Holmes, R., Potokina, M., Bogacheva, J., James, M., & Sullivan, B. (2025). PyEarthTools: Machine learning for Earth system science (0.5.1). Zenodo. https://doi.org/10.5281/zenodo.17544431
::::
::::{tab-item} BibTeX
```
@software{leeuwenburg_2025_17544431,
  author       = {Leeuwenburg, Tennessee and
                  Cook, Harrison and
                  Rio, Maxime and
                  Hobeichi, Sanaa and
                  Miller, Joel and
                  Mason, Gemma and
                  Ramanathan, Nikeeth and
                  Pill, John and
                  Haddad, Stephen and
                  Stassen, Christian and
                  de Burgh-Day, Catherine and
                  Holmes, Ryan and
                  Potokina, Margarita and
                  Bogacheva, Jenya and
                  James, Matthew and
                  Sullivan, Ben},
  title        = {PyEarthTools: Machine learning for Earth system
                   science
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.5.1},
  doi          = {10.5281/zenodo.17544431},
  url          = {https://doi.org/10.5281/zenodo.17544431},
}
```
::::
:::::






```{toctree}
:hidden:
:caption: 'Index to Documentation:'
:maxdepth: 2

self
newuser
newproject
projectideas
installation
notebooks/Gallery
data
config
api/api
roadmap
contributing
devguide
maintainer
```
