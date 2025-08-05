# Pipeline API How-To Guide

A pipeline is a Python iterator with the job of supplying input/output pairs to a machine learning framework, or to a PyEarthTools registered model interface.

It is somewhat similar to an IterableDataset in PyTorch, or a DataLoader in PyTorch Lightning. However, it can be used with not only these frameworks, but many others as well. The advantages of using a PyEarthTools pipeline are:

- More modular and flexible if you want to use the same pipeline with multiple models
- Capable of supplying data to PyTorch, XGBoost, MLX, Tensorflow and JAX
- Includes pre-coded processing steps for many common operations

A how-to guide is coming soon. In the meantime, see the pipeline tutorials in the tutorial gallery.
