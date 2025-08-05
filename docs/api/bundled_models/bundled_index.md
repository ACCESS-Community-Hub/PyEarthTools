# Bundled Models Index

Unlike the other directories in the 'packages' directory of PyEarthTools, the "bundled_models" directory does not itself contain a "bundled models" Python package. Rather, it contains multiple model packages in separate directories. Each of these bundled models **is** a Python package. As such, "bundled_models" is not itself installable. This page will provide an index table for each bundled model. At the current time, FourCastNeXt is the only bundled model.

Bundled models also have configuration files in addition to the the Python code. Each yaml file is also included in the table for the bundled model.

The API docs for each bundled model will also be presented together in the [Bundled Models API Docs](bundled_api.md).

## FourCastNeXt Bundled Model

|  Module                        |       Purpose                               |   API Docs     |
|--------------------------------|---------------------------------------------|----------------|
|  `fourcastnext`                |  PyTorch Lightning API                      | - [lightning_model.FourCastNextML](bundled_api.md#fourcastnext.lightning_model.FourCastNextLM)  |
|                                |  PyEarthTools Registration Interface        | - [registered_model](bundled_api.md#fourcastnext.registered_model.FourCastNextRM) |
|                                |  Crop ERA5 grid to required spacing         | - [CropToRectangle](bundled_api.md#fourcastnext.CropToRectangle) |
|                                |  Crop ERA5 low res grid to required spacing | - [CropToRectangleSmall](bundled_api.md#fourcastnext.CropToRectangleSmall) |
|  `fourcastnext.architecture`   |  Multilayer Perceptron                      | - [Mlp](bundled_api.md#fourcastnext.architecture.Mlp)  |
|                                |  2D AFNO Network                            | - [AFNO2D](bundled_api.md#fourcastnext.architecture.AFNO2D)  |
|                                |                                             | - [Block](bundled_api.md#fourcastnext.architecture.Block)  |
|                                |                                             | - [AFNONet](bundled_api.md#fourcastnext.architecture.AFNONet)  |
|                                |  Patching and embedding                     | - [PatchEmbed](bundled_api.md#fourcastnext.architecture.PatchEmbed)  |
|  Training Directory            |  Configuration for different experiments    |                 |
|  training/configs              |  FourCastNeXt default configuration         | config.yaml   |
|                                |  Worker and batch size for data preprocessing | data/module/default.yaml   |
|                                |  Default model data split                   | data/splits/default.yaml   |
|                                |  Data patch size and number of channels     | data/example.yaml   |
|                                |  Train on a reduced data set                | splits/short_training.yaml   |
|                                |  PyTorch model initialisation parameters    | model/default.yaml   |
|                                |  Training strategy configuration            | trainer/default.yaml   |
|  training/limited_variables_early_stopping |  FourCastNeXt full-size reduced-training configuration | limited_vars_early_stop.yaml   |
|                                |  FourCastNeXt low resolution configuration  | lowres.yaml   |
|                                |  Worker and batch size for training         | module/default.yaml   |
|                                |  Full-length training period                | splits/default.yaml   |
|                                |  Reduced-length training period             | splits/short_training_splits.yaml   |
|                                |  Reduced-length training period             | splits/short_training_splits.yaml   |
|                                |  Full-res data and channels                 | data/example.yaml   |
|                                |  Low-res data and channels                  | data/lowres.yaml   |
|                                |  PyTorch model initialisation parameters    | model/default.yaml   |
|                                |  Training strategy for full convergence     | trainer/default.yaml   |
|                                |  Train a reduced number of epocs            | trainer/few_epochs.yaml   |
|  Pipelines Directory           |  Define data normalisation pipeline         |                 |
|  pipelines                     |  Full-resolution pipeline                   | early_stopping.pipe   |
|                                |  Low-resolution pipeline                    | low_res_demo_subset.pipe   |
|                                |  General random data for testing            | example.pipe   |
