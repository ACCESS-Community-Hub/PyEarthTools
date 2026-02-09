# Persistence Model for use with the PyEarthTools Package

**TODO: description**

## Installation

Clone the repository, then run
```shell
pip install -e .
```

## Training

No training is required for this model. It computes persistence on-the-fly using historical data loaded via the PET pipeline.

## Predictions / Inference

You can generate persistence values out of the box using the `pet predict` command line API, or by using a Jupyter Notebook as demonstrated in the tutorial gallery.

```shell
pet predict
```

and `Development/Persistence` should be visible.

If so, you can now run some inference.

```shell
pet predict --model Development/Persistence
```

When running the command, it will prompt for other required arguments.

**TODO: description of required arguments**


#### Example

```shell
pet predict --model Development/Persistence # TODO
```

## Acknowledgments

Not applicable. Heuristically developed.
