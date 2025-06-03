# FourCastNeXt Model for use with the PyEarthTools Package

For more detail regarding the FourCastNeXt model, see [Guo et al. (2024)](https://doi.org/10.48550/arXiv.2401.05584).

This package allows PyEarthTools users to utilise the neural network architecture employed by the FourCastNeXt model. It also allows the architecture to be applied to other data, and trained using different strategies. It does not include pre-trained weights. The example training code does uses a simplified training strategy which does not reproduce the paper.

## Installation

Clone the repository, then run
```shell
pip install -e .
```

## Training

Once data has been cached to disk, training can be run with `fourcastnext/Training/train.py`. This contains a python script to run, and a dgxa100 compute job. There is also a Jupyter Notebook which can be used to run the training.

Additionally, you can change

- Save path
- Batch size
- Number of workers

## Predictions / Inference

If you have successfully run the training, you can now run some predictions either using the `pet predict` command line API, or by using a Jupyter Notebook as demonstrated in the tutorial gallery.

```shell
pet predict
```

and `Development/FourCastNeXt` should be visible.

If so, you can now run some inference.

```shell
pet predict --model Development/FourCastNeXt
```

When running the command, it will prompt for other required arguments.

Set `ckpt_path` to the full path of the checkpoint of the model you wish to load. It will then be copied to your assets folder and loaded.

#### Example

```shell
pet predict --model Development/FourCastNeXt --ckpt_path PATH_TO_CHECKPOINT
```

## Acknowledgments

This package extends and is significantly based on the code from https://github.com/nci/FourCastNeXt which is made available
under the Apache 2.0 license. That repository in turn extends the code from https://github.com/NVlabs/FourCastNet/, released under the BSD 3-Clause license.
The FourCastNet model is described in detail at https://doi.org/10.48550/arXiv.2202.11214. The FourCastNeXt model is described in detail at https://doi.org/10.48550/arXiv.2401.05584,
and a version of the FourCastNeXt code is bundled, adapted for compatibility and maintained within the PyEarthTools repository so it can continue to be a useful
reference implementation and learning aid.
