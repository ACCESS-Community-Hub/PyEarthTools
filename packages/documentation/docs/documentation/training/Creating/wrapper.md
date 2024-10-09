# Wrapper's

As explained in [training](./../index.md) a `ModelWrapper` connects an ML model to a `PipelineDataModule`.

A subclass is expected to implement the basic functionality of `saving`, `loading`, and `predicting`.
The super class of `ModelWrapper` handles the conversion of data to datamodules if not given as one, which the type of which can be configured by `_default_datamodule`. Additionally, the model passed to the root class can be accessed at `,model`. 

!!! Warning:
    It is worth noting that the model will not be recorded in the initialisation by default due to the typical unyamlable nature of complex models. 

    This behaviour can be changed by setting `_record_model` to `True`.

## Training Wrapper

If a framework enables training, i.e. is not just a predictor, it can inherit from `TrainingWrapper` which expects an implementation of `fit`.

## Implementation

The actual implementation of the wrapper is up to the user. 