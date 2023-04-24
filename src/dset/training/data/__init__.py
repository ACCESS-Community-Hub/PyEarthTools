"""
`dset.training` Data Pipeline Creation

With [dset.training.data][dset.training.data] it is possible to create easily expandable and configurable data pipeline to prepare data for an ML Model.


# Examples

### Example 1 
Load ERA5, grab two samples and patch it into `64` by `64` arrays.
=== "Yaml File"
    ```yaml
    ## Data Pipeline Configuration
    data:
        ## Pipeline
        Source:
            ## For more complicated data sources, 
            ## `indexes.InterpolationIndex` can be used to interpolate together multiple DataIndexes
            archive.ERA5:
                variables: ['2t']
                level: 'single'
            ## Apply dset.data.transforms
            ## Retrieve 1 before and 1 after, at 60 min interval
            iterators.TemporalInterface:
                samples : [1,1]
                sample_interval : [60, 'minutes']
            ## Iterate
            iterators.Iterator:
                catch: ['dset.data.DataNotFoundError', 'ValueError', 'OSError']
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
            loaders.PytorchIterable: {}

    ```

=== "Python Code"
    ```python
    import dset.training
    import dset.data

    ## ERA5 Loader
    ERA5 = dset.data.archive.ERA5(variables = ['2t'], level = 'single')

    ## Data Pipeline
    ### Retrieve 1 before and 1 after, at 60 min interval
    datapipe = dset.training.data.iterators.TemporalInterface(ERA5, samples = [1,1], sample_interval = [60, 'minutes'])
    ### Iterate 
    datapipe = dset.training.data.iterators.Iterator(datapipe, catch = ['dset.data.DataNotFoundError', 'ValueError', 'OSError'])
    ### Drop Data that is all nan's
    datapipe = dset.training.data.operations.filters.DropAllNan(datapipe)
    ### Patch into 64 by 64 arrays
    datapipe = dset.training.data.operations.PatchingDataIndex(datapipe, kernel_size = [64,64])
    ### Fill all nan's with 0
    datapipe = dset.training.data.operations.values.FillNa(datapipe)
    ### Drop data with more than 50% 0's
    datapipe = dset.training.data.operations.filters.DropValue(datapipe, value = 0,  percentage= 50)
    ### Ensure no nan's
    datapipe = dset.training.data.operations.filters.DropNan(datapipe)
    ### Rearrange axis
    datapipe = dset.training.data.operations.reshape.Rearrange(datapipe, rearrange = 'c t h w -> t c h w')

    ```


"""
from dset.training.data.templates import (
    DataIterator,
    DataOperation,
    DataIterationOperator,
    DataInterface,
)
from dset.training.data.sequential import SequentialIterator, Sequential, from_dict
from dset.training.data import interfaces, iterators, operations, indexes
from dset.training.data import sanity
