# DSET Training

Using `dset.data` this package aims to significantly speed up ML model investigations and experimentations. 

Using `dset.data.DataIndex` a data source can be specified and through `dset.training.data` a data pipeline can be built to normalise, reshape, and filter incoming training data.

Afterwhich a model can be specified and using [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) trained on any system with as many GPU's as needed.



## Installing
---
This repo can be cloned and installed, or loaded as a module on GADI.
``` shell
module use /g/data/kd24/modulefiles/

module load dset
#or
module load dset/training
```

## Dependencies
`dset.training` has a number of dependencies.

- `dset.data`
- [Torch](https://pytorch.org/)
- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) 
- [einops](https://github.com/arogozhnikov/einops) 



## Command Line
`dset.training` includes a command line interface for user convenience, `training`
```shell
> training
Usage: training [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  fit      From Yaml Config Fit Model
  predict  Using Yaml Config & Checkpoint, predict at index
```

Currently, a fit and predict command have been created