# PyEarthTools: Reproducible machine learning for Earth system science

PyEarthTools is a Python framework, containing modules for loading data; pre-processing, normalising and standardising data; defining machine learning (ML) models; training ML models; performing inference with ML models; and evaluating ML models. It contains specialised support for weather and climate data sources and models. It has an emphasis on reproducibility, shareable pipelines, and human-readable low-code pipeline definition.

Source Code: [github.com/ACCESS-Community-Hub/PyEarthTools](https://github.com/ACCESS-Community-Hub/PyEarthTools)  
Documentation: [pyearthtools.readthedocs.io](https://pyearthtools.readthedocs.io)  
Tutorial Gallery: [available here](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html)


> [!NOTE]
> **THIS REPOSITORY IS UNDER CONSTRUCTION**
>
> This repository contains code which is under construction, and should not yet be used outside of a research setting.
> The development team are working busily to bring everything up to spec. As such, things are likely
> to change pretty often. Please take a look around!
>

PyEarthTools comprises multiple sub-packages which can be used individually or together.

|  Sub-Package |  Purpose  |
|--------------|---------------------- |
|  Data        | Loading and indexing into well-known Earth system data sets to produce ML-ready data structures |
|  Utils       | Code for common functionality across the sub-packages | 
|  Pipeline    | Definining reproducible sequences of operations with the ability to cache results |
|  Training    | Code defining the training processes and schedules of a machine learning model |
|  Tutorial    | Contains helper code for data data sets used in tutorials |
|  Models      | (Coming soon) Contains code for defining model architectures  |
|  Registry    | (Coming soon) Contains specific, bundled models which can be easily run |
|  Evaluation  | (Coming soon) Contains code for producing standard evaluations (such as benchmarks and scorecards)

[![Coverage Status](https://coveralls.io/repos/github/ACCESS-Community-Hub/PyEarthTools/badge.svg)](https://coveralls.io/github/ACCESS-Community-Hub/PyEarthTools) <-- we are working towards 100% test coverage of PyEarthTools code

# Overview of documentation

We have information for:

 - [New user guides and introduction to the concepts in PyEarthTools](https://pyearthtools.readthedocs.io/en/latest/newuser.html)
 - [Installation instructions](https://pyearthtools.readthedocs.io/en/latest/installation.html) for different usage scenarios
 - [Data catalogue setup](https://pyearthtools.readthedocs.io/en/latest/catalogue.html) for facility managers or individuals to establish their research data catalogue
 - [A tutorial gallery with a wide variety of examples](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html)
 - Much more, including how-to guides, project setup guide, information on accessing data, guides to evaluation, orientiation for
   physical scientists and data scientists
