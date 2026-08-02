# PyEarthTools: Machine learning for Earth system science

[![DOI](https://zenodo.org/badge/903938118.svg)](https://doi.org/10.5281/zenodo.15760768)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FACCESS-Community-Hub%2FPyEarthTools%2Frefs%2Fheads%2Fdevelop%2Fpyproject.toml&label=Python)

- A Python framework that supports the develoment of machine learning models, big and small, for Earth system science
- Suitable for students and newcomers, as well as for domain specialists and scientists
- Runs effectively on HPC (supercomputers), cloud, workstations and laptops
- Suitable for use with megabytes to petabytes of data
- Still under early-stage development - things are likely to change a lot. If you notice an issue, please feel free to raise it on GitHub

|![](https://pyearthtools.readthedocs.io/en/latest/_images/notebooks_demo_FourCastNeXt_Inference_9_1.png)<br>A weather prediction from a model trained with PyEarthTools.|![](https://pyearthtools.readthedocs.io/en/latest/_images/notebooks_tutorial_Working_with_Climate_Data_14_2.svg)<br>A data processing flow composed for working with climate data.|
|:-:|:-:|

Source Code: [github.com/ACCESS-Community-Hub/PyEarthTools](https://github.com/ACCESS-Community-Hub/PyEarthTools)  
Documentation: [pyearthtools.readthedocs.io](https://pyearthtools.readthedocs.io)  
Tutorial Gallery: [available here](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html)  
New Users Guide: [available here](https://pyearthtools.readthedocs.io/en/latest/newuser.html)  

**If you use `PyEarthTools` for your work or a publication, [please cite our work](https://pyearthtools.readthedocs.io/en/latest/#citing-pyearthtools).**

## Installation

**Here is the quickest way to install the complete framework and get started:**

We strongly recommend using either a Conda or Python [virtual environment](https://pyearthtools.readthedocs.io/en/latest/installation.html#virtual-environments).

Run the following commands to install PyEarthTools in a Conda environment:
```shell
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
cd PyEarthTools
conda create -y -p ./venv python=3.13 graphviz
conda activate ./venv
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Alternatively, run the following commands to install PyEarthTools in a Python virtual environment:
```shell
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab
```
> [!TIP]
> (Optional) Install [Graphviz](https://graphviz.org/download/) (not installable via pip) to display pipelines.

PyEarthTools comprises multiple sub-packages which may be installed and used separately. See the [installation guide](https://pyearthtools.readthedocs.io/en/latest/installation.html) for more details.

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

|    Sub-Package                 |  Purpose  |
|--------------------------------|---------------------- |
|  [Data](https://pyearthtools.readthedocs.io/en/latest/api/data/data_index.html)    | Loading and indexing Earth system data into xarray |
|  [Utils](https://pyearthtools.readthedocs.io/en/latest/api/utils/utils_index.html)  | Code for common functionality across the sub-packages |
|  [Pipeline](https://pyearthtools.readthedocs.io/en/latest/api/pipeline/pipeline_index.html)       |  Process and normalise Earth system data ready for machine learning |
|  [Training](https://pyearthtools.readthedocs.io/en/latest/api/training/training_index.html)       | Training processes for machine learning models |
|  [Tutorial](https://pyearthtools.readthedocs.io/en/latest/api/tutorial/tutorial_index.html)       | Contains helper code for data sets used in tutorials |
|  [Bundled Models](https://pyearthtools.readthedocs.io/en/latest/api/bundled_models/bundled_index.html) | Maintained versions of specific, bundled models which can be easily trained and run |
|  [Zoo](https://pyearthtools.readthedocs.io/en/latest/api/zoo/zoo_index.html)            | Contains code for managing registered models (such as the bundled models) |
|  Evaluation     | (Coming soon) Contains code for producing standard evaluations (such as benchmarks and scorecards) |

## Citing `PyEarthTools`

If you use PyEarthTools for your work, we would appreciate you citing our software as below:

Leeuwenburg, T., Cook, H., Rio, M., Hobeichi, S., Miller, J., Mason, G., Ramanathan, N., Pill, J., Haddad, S., Stassen, C., de Burgh-Day, C., Holmes, R., Potokina, M., Bogacheva, J., James, M., & Sullivan, B. (2025). PyEarthTools: Machine learning for Earth system science (0.5.1). Zenodo. https://doi.org/10.5281/zenodo.17544431

BibTeX:
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
