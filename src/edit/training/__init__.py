"""
EDIT Training 

Using [edit.data][edit.data.indexes] DataIndexes prepare data for training, 
and allow rapid distributed training of Machine Learning Models.

## Sections
| Name | Description |
| ---- | --------------------- |   
| data | Data Preparation and Loading |
| modules | Ancillary Machine Learning Modules |
| trainer | Training Classes and Objects |


## Example

Load ERA5, and feed it into a model
=== "Yaml File"
    ```yaml
    model:
        Source: 'Models.Architecture'
        model_params:
            img_size: 256
            in_channels: 1
            out_channels: 1

    data:
        Source:
            ## For more complicated data sources, 
            ## `indexes.InterpolationIndex` can be used to interpolate together multiple DataIndexes
            archive.ERA5:
                variables: ['2t']
                level: 'single'
            ## Apply edit.data.transforms
            ## Retrieve 1 before and 1 after, at 60 min interval
            iterators.TemporalInterface:
                samples : [1,1]
                sample_interval : [60, 'minutes']
            ## Iterate
            iterators.Iterator:
                catch: ['edit.data.DataNotFoundError', 'ValueError', 'OSError']
            ## Drop Data that is all nan's
            operations.filters.DropAllNan: {}
            ## Patch into 64 by 64 arrays
            operations.PatchingDataIndex:
                kernel_size: [64,64]
            ## Fill all nan's with 0
            operations.values.FillNa:
                apply_iterator: True
            ## Drop data with more than 50% 0's
            operations.filters.DropValue:
                value: 0
                percentage: 50
            ## Ensure no nan's
            operations.filters.DropNan: {}
            ## Rearrange axis
            operations.reshape.Rearrange:
                rearrange: 'c t h w -> t c h w'
            ## Connect with Pytorch Iterables
            loader.PytorchIterable: {}

        Ranges:
            train_data:
                start: '2021-01-01T00:00'
                end: '2022-01-01'
                interval: 60
            valid_data:
                start: '2022-01-01T00:00'
                end: '2022-04-01'
                interval: 10
    trainer:
        root_dir: '/models/'
        num_workers: 12
        strategy: 'ddp'
        batch_size: 64

    ```

=== "Python Code"
    ```python
    import edit.training
    import edit.data

    ## ERA5 Loader
    ERA5 = edit.data.archive.ERA5(variables = ['2t'], level = 'single')

    ## Data Pipeline
    ### Retrieve 1 before and 1 after, at 60 min interval
    datapipe = edit.pipeline.iterators.TemporalInterface(ERA5, samples = [1,1], sample_interval = [60, 'minutes'])
    ### Iterate 
    datapipe = edit.pipeline.iterators.Iterator(datapipe, catch = ['edit.data.DataNotFoundError', 'ValueError', 'OSError'])
    ### Drop Data that is all nan's
    datapipe = edit.pipeline.operations.filters.DropAllNan(datapipe)
    ### Patch into 64 by 64 arrays
    datapipe = edit.pipeline.operations.PatchingDataIndex(datapipe, kernel_size = [64,64])
    ### Fill all nan's with 0
    datapipe = edit.pipeline.operations.values.FillNa(datapipe)
    ### Drop data with more than 50% 0's
    datapipe = edit.pipeline.operations.filters.DropValue(datapipe, value = 0,  percentage= 50)
    ### Ensure no nan's
    datapipe = edit.pipeline.operations.filters.DropNan(datapipe)
    ### Rearrange axis
    datapipe = edit.pipeline.operations.reshape.Rearrange(datapipe, rearrange = 'c t h w -> t c h w')
    ### Connect to PyTorch Iterable
    datapipe = edit.training.loader.PytorchIterable(datapipe)

    ## Model
    import Models.Architecture
    model = Models.Architecture(model_params = dict(img_size = 256, in_channels = 1, out_channels = 1))

    ## Trainer
    trainer = edit.training.trainer.EDITTrainerWrapper(model = model, train_data = datapipe, root_dir = '/models/', num_workers = 12, strategy = 'ddp', batch_size = 64)


    ```
    

"""

from edit.training import models, modules, trainer, loader
from edit.training.trainer import EDITTrainer, EDITLightningTrainer, from_yaml, MLDataIndex

from_dict = from_yaml
load = from_yaml

if __name__ == "__main__":
    trainer.commands.entry_point()
