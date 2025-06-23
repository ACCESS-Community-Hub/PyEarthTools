% scores documentation master file, created by
% sphinx-quickstart on Sat Sep  9 11:24:53 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# PyEarthTools: Machine learning for Earth system science

- An approachable way for researchers to get started with ML research for Earth system science
- Provides a software framework for research and experimentation
- Also suitable for students and newcomers
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

## Installation

**Here is the quickest way to install the complete framework and get started:**

From within a suitable [virtual environment](installation.md#virtual-environments), run the following commands:
```
git clone git@github.com:ACCESS-Community-Hub/PyEarthTools.git
pip install -r requirements-dev.txt
conda install graphviz
cd notebooks
jupyter lab
```

PyEarthTools comprises multiple sub-packages which may be installed and used separately. See the [installation guide](installation.md) for more details.

## Overview of PyEarthTools

PyEarthTools is a Python framework containing modules for:
 - loading and fetching data; 
 - pre-processing, normalising and standardising data into a normal form suitable for machine learning; 
 - defining machine learning (ML) models; 
 - training ML models and managing experiments;
 - performing inference with ML models; 
 - and evaluating ML models. 

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

## Acknowleging or Citing `PyEarthTools`

If you use PyEarthTools for your work, we would appreciate you acknowledging our work. A citable DOI will be available soon. In the meantime, please cite this repository.
  

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
config
api/api
roadmap
devguide
maintainer
```
