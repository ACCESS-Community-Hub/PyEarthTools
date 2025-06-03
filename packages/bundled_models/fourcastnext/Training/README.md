## Training FourCastNeXt

Using OmegaConf and Hydra, the configuration of the model and the data can be broken down into nested configuration files.

These can then be selected from at the command line and used independently.

```shell
python train.py data=example
```

## Example Full config

This is the full config made from

`python python train.py data=example`

```yaml

experiment_name: ${now:%Y-%m-%d}/${now:%H-%M-%S}
path: /scratch/kd24/${oc.env:USER}/ML/FourCastNeXt/Training/${experiment_name}
fit: false
trainer:
  precision: 16-mixed
  max_epochs: 200
  checkpointing:
  - monitor: train_loss
    mode: min
    dirpath: '{path}/Checkpoints/Train'
    filename: model-{epoch:02d}-{step:02d}
    every_n_train_steps: 1000
  - monitor: valid_loss
    mode: min
    dirpath: '{path}/Checkpoints/Valid'
    filename: model-{epoch:02d}-{step:02d}-{valid_loss}
    every_n_train_steps: 5000
  - monitor: epoch
    mode: max
    dirpath: '{path}/Checkpoints/Epoch'
    filename: model-{epoch:02d}
    save_on_train_epoch_end: true
    save_top_k: 50
model:
  _target_: fourcastnext.model.FourCastNext
  _recursive_: true
  model_params:
    img_size: ${data.img_size}
    in_channels: ${data.in_channels}
    out_channels: ${data.out_channels}
data:
  pipelines:
  - pipelines/example.pipe
  img_size:
  - 720
  - 1440
  in_channels: 2
  out_channels: 2
  splits:
    random: true
    random_seed: 42
    train:
    - 1980
    - 2018
    - 6 hours
    valid:
    - 2018
    - 2020
    - 6 hours
  module:
    num_workers: 12
    batch_size: 8
```
