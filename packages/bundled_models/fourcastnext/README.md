# FourCastNeXt Model for use with the PyEarthTools Package

For more detail regarding the FourCastNeXt model, see [Guo et al. (2024)](https://doi.org/10.48550/arXiv.2401.05584).

## Installation

Clone the repository, then run
```shell
pip install -e .
```

## Training

Once data has been cached to disk, training can be run with `fourcastnext/Training/train.py`. This contains a python script to run, and a dgxa100 compute job.

If you have changed the pipeline, ensure you update it in the script.

Additionally, you can change

- Save path
- Batch size
- Number of workers

## Predictions / Inference

If you have successfully run the training, you can now run some predictions with `pyearthtools.zoo`.

Again, if you have changed the pipeline, you will need to create the configs necessary for inference.

Simply these configs are the original pipeline broken into data retrieval and then pipeline operations with the removal of caches.

### Setup

Set the config path

```shell
export PYEARTHTOOLS_MODELS_CONFIGS=PATH_TO_CONFIGS:$PYEARTHTOOLS_MODELS_CONFIGS
```

Set the dynamic import

```shell
export PYEARTHTOOLS_MODELS_IMPORTS='fourcastnext'@PATH_TO_FOURCASTNEXT/src
```

Once those have been set you should be able to run

```shell
pet models
```

and `Development/FourCastNeXt` should be visible.

If so, you can now run some inference.

```shell
pet interactive --model Development/FourCastNeXt
```

When running the command, it will prompt for other kwargs (which fyi could be included in the initial command call),

Set `ckpt_path` to the full path of the checkpoint of the model you wish to load. It will then be copied to your asset folder and loaded

#### Example

```shell
pet interactive --model Development/FourCastNeXt --ckpt_path PATH_TO_CHECKPOINT
```

## Acknowledgments

This package extends and is significantly based on the code from https://github.com/nci/FourCastNeXt which is made available
under the Apache 2.0 license. That repository in turn extends the code from https://github.com/NVlabs/FourCastNet/, released under the BSD 3-Clause license.
The FourCastNet model is described in detail at https://doi.org/10.48550/arXiv.2202.11214. The FourCastNeXt model is described in detail at https://doi.org/10.48550/arXiv.2401.05584,
and a version of the FourCastNeXt code is bundled, adapted for compatibility and maintained within the PyEarthTools repository so it can continue to be a useful
reference implementation and learning aid.
