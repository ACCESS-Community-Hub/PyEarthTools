# EDIT Training

Using `edit.data` this package aims to significantly speed up ML model investigations and experimentations. 

Using `edit.data.DataIndex` a data source can be specified and through `edit.pipeline` a data pipeline can be built to normalise, reshape, and filter incoming training data.

Afterwhich a model can be specified and using [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) trained on any system with as many GPU's as needed.
