# Pipeline API How-To Guide

A pipeline is a [Python iterator](https://wiki.python.org/moin/Iterator) with the job of supplying input/output pairs to a machine learning framework, or to a PyEarthTools registered model interface.

It is somewhat similar to an IterableDataset in PyTorch, or a DataLoader in PyTorch Lightning. However, it can be used with not only these frameworks, but many others as well. The advantages of using a PyEarthTools pipeline are:

- More modular and flexible if you want to use the same pipeline with multiple models
- Capable of supplying data to PyTorch, XGBoost, MLX, Tensorflow and JAX
- Includes pre-coded processing steps for many common operations

For more information, please see:

- [Introduction to data pipelines](project:/notebooks/tutorial/Data_Pipelines.ipynb) 
- [Working with Multiple Data Sources](project:/notebooks/tutorial/MultipleSources.ipynb)
- [The pipeline API tutorials](project:/notebooks/Gallery.md#Deep-Dive---The-Pipeline-Module) in the tutorial gallery
- The [pyearthtools.pipeline](pipeline_index.md) API documentation index
