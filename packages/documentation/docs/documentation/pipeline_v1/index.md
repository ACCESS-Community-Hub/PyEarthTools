# EDIT.pipeline - DEPRECATED

`edit.pipeline` Data Pipeline Creation

With [edit.pipeline][edit.pipeline] it is possible to create easily expandable and configurable data pipeline to prepare data for an ML Model.

Ultimately using [edit.pipeline][edit.pipeline] is about ordering and constructing the pipeline from either the available and predeveloped steps, or those which can be created by the user.

## Examples

Load ERA5, grab four samples and patch it into `64` by `64` arrays

### Code

=== "Yaml File"
    ```yaml
    ## Data Pipeline Configuration
    data:
        ## For more complicated data sources, 
        ## `indexes.InterpolationIndex` can be used to interpolate together multiple DataIndexes
        archive.ERA5:
            variables: ['2t']
            level: 'single'
        ## Retrieve 1 before and 1 after, at 60 min interval
        iterators.TemporalInterface:
            samples : [2,2]
            sample_interval : [60, 'minutes']
        ## Iterate
        iterators.Iterator:
            catch: ['edit.data.DataNotFoundError', 'ValueError', 'OSError']
        ## Drop Data that is all nan's
        operations.filter.DropAllNan: {}
        ## Patch into 64 by 64 arrays
        operations.Patch:
            kernel_size: [64,64]
        ## Fill all nan's with 0
        operations.value.FillNa:
            apply_iterator: True
        ## Drop data with more than 50% 0's
        operations.filter.DropValue:
            value: 0
            percentage: 50
        ## Ensure no nan's
        operations.filter.DropNan: {}
        ## Rearrange axis
        operations.reshape.Rearrange:
            rearrange: 'c t h w -> t c h w'
        ## Connect with Pytorch Iterables
        loaders.PytorchIterable: {}

    ```

=== "Python Code"
    ```python
    import edit.pipeline
    import edit.data

    ## ERA5 Loader
    ERA5 = edit.data.archive.ERA5(variables = ['2t'], level = 'single')

    ## Data Pipeline
    ### Retrieve 2 before and 2 after, at 60 min interval
    datapipe = edit.pipeline.iterators.TemporalInterface(ERA5, samples = [2,2], sample_interval = [60, 'minutes'])
    ### Iterate 
    datapipe = edit.pipeline.iterators.Iterator(datapipe, catch = ['edit.data.DataNotFoundError', 'ValueError', 'OSError'])
    ### Drop Data that is all nan's
    datapipe = edit.pipeline.operations.filter.DropAllNan(datapipe)
    ### Patch into 64 by 64 arrays
    datapipe = edit.pipeline.operations.Patch(datapipe, kernel_size = [64,64])
    ### Fill all nan's with 0
    datapipe = edit.pipeline.operations.value.FillNa(datapipe)
    ### Drop data with more than 50% 0's
    datapipe = edit.pipeline.operations.filter.DropValue(datapipe, value = 0,  percentage= 50)
    ### Ensure no nan's
    datapipe = edit.pipeline.operations.filter.DropNan(datapipe)
    ### Rearrange axis
    datapipe = edit.pipeline.operations.reshape.Rearrange(datapipe, rearrange = 'c t h w -> t c h w')

    ```

=== "Sequential Python Code"
    ```python
    import edit.pipeline
    import edit.data

    edit.pipeline.Sequential(
        ## ERA5 Loader
        edit.data.archive.ERA5(variables = ['2t'], level = 'single'),
        ### Retrieve 2 before and 2 after, at 60 min interval
        edit.pipeline.iterators.TemporalInterface(samples = [2,2], sample_interval = [60, 'minutes']),
        ### Iterate 
        edit.pipeline.iterators.Iterator(catch = ['edit.data.DataNotFoundError', 'ValueError', 'OSError']),
        ### Drop Data that is all nan's
        edit.pipeline.operations.filter.DropAllNan(),
        ### Patch into 64 by 64 arrays
        edit.pipeline.operations.Patch(kernel_size = [64,64]),
        ### Fill all nan's with 0
        edit.pipeline.operations.value.FillNa(),
        ### Drop data with more than 50% 0's
        edit.pipeline.operations.filter.DropValue(value = 0,  percentage= 50),
        ### Ensure no nan's
        edit.pipeline.operations.filter.DropNan(),
        ### Rearrange axis
        edit.pipeline.operations.reshape.Rearrange(rearrange = 'c t h w -> t c h w'),
    )
    ```

### Diagram

!!! info inline end "Diagram"
    ```mermaid

        flowchart TD
            A[ERA5] -->|1,1,721,1440| B[TemporalIterator]
            B -->|2,1,2,721,1440| C[DropAllNan]
            C -->|2,1,2,721,1440| D[Patch]
            D -->|2,276,1,2,64,64| E[FillNa & DropValue & DropNan]
            E -->|2,276,1,2,64,64| F[Rearrange]
            F -->|2,276,2,1,64,64| G[ML Model]

    ```

This diagram shows the above pipelines, with the shape of the data indicated between steps
