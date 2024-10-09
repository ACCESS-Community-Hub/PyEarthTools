# Pipeline Catalog

`edit.pipeline` contains a number of predeveloped pipeline steps whihc can be used to config data and prepare it for any downstream applications.

The pipeline creation can be likened to a puzzle, with the primary challenge being combining steps together to achieve the goal.

Currently, the pipeline is run sequentially, with each step taking the prior step as it's first `__init__` argument, however, each step
is configured to allow partial initialisation, as shown in [loading](/documentation/pipeline/loading).


!!! Hint
    All operations part of `edit.pipeline` which chnage the structure of data implement and `undo` function,
    thus, data can be restored to it's original state, or at least close to it.

    ```python

    forward_data  = pipeline(date)
    original_data = pipeline.undo(forward_data)

    ```

## Operations

This page seeks to list all steps developed, and when to use them. All paths shown below are relative to `edit.pipeline.operations`.

### Augmentation

| Name | Path | Purpose |
| ---- | ---- | ------: |
| Rotate    | `augmentation.Rotate`    | Rotate data by 90 &deg randomly    |
| Flip      | `augmentation.Flip`      | Flip data on axis randomly         |
| Transform | `augmentation.Transform` | Combine Flip & Rotate augmentation |

### Filters

| Name | Path | Purpose |
| ---- | ---- | ------: |
| DropAnyNan | `filter.DropAnyNan` | Drop data with any nans |
| DropAllNan | `filter.DropAllNan` | Drop data which all nans |
| DropValue  | `filter.DropValue`  | Drop data which contains more than a given percentage of a value |
| Shape      | `filter.Shape`      | Drop data which is not the correct shape |

### Padding

| Name | Path | Purpose |
| ---- | ---- | ------: |
| UndoPadder | `pad.UndoPadder` | Pad data to the correct shape when undoing |

### Patching

| Name | Path | Purpose |
| ---- | ---- | ------: |
| Patch | `Patch` | Patch a dataset or array into smaller arrays |

### Reshaping

| Name | Path | Purpose |
| ---- | ---- | ------: |
| Rearrange  | `reshape.Rearrange`  | Rearrange dimensions of an array with `einops` notation |
| Squish     | `reshape.Squish`     | Squish one dimensional axis |
| Expand     | `reshape.Expand`     | Expand dimension of an array |
| Flatten    | `reshape.Flatten`    | Flatten data along given axis |
| Dimensions | `reshape.Dimensions` | Reorder xarray dimensions |

### Sampling

| Name | Path | Purpose |
| ---- | ---- | ------: |
| RandomSampler  | `sample.RandomSampler` | Randomly sample from a buffer |
| RandomDropOut  | `sample.RandomDropOut` | Randomly drop out data |
| DropOut        | `sample.DropOut`       | Drop out data at an interval |

### Selecting

| Name | Path | Purpose |
| ---- | ---- | ------: |
| Select         | `select.Select` | Select an element from an array |
| SelectDataset  | `sample.SelectDataset`  | Select variables from an xarray dataset |

### Sorting

| Name | Path | Purpose |
| ---- | ---- | ------: |
| xarraySorter   | `sort.xarraySorter` | Sort xarray variables |

### Transforms

| Name | Path | Purpose |
| ---- | ---- | ------: |
| TransformOperation   | `TransformOperation` | Apply transforms to data |

Specific transforms, all paths relative to `edit.pipeline.operations.transforms`

| Name | Path | Purpose |
| ---- | ---- | ------: |
| time_of_year   | `time_of_year` | Add time of year as variable to xarray dataset |

### ToNumpy

| Name | Path | Purpose |
| ---- | ---- | ------: |
| ToNumpy  | `ToNumpy` | Convert xarray dataset to a numpy array |

### Values

| Name | Path | Purpose |
| ---- | ---- | ------: |
| FillNan   | `value.FillNan` | Fill nans with a value |
| MaskValue  | `value.MaskValue` | Replace a value based on a condition  |
| ForceNormalised  | `value.ForceNormalised` | Force data to be between a min and max  |

## Indexes

All paths shown below are relative to `edit.pipeline`.

| Name | Path | Purpose |
| ---- | ---- | ------: |
| InterpolationIndex    | `indexes.InterpolationIndex`    | Interpolate Indexes together |
| MergeIndex            | `indexes.MergeIndex`      | Merge Indexes together         |
| CoordinateIndex       | `indexes.CoordinateIndex` | Add coordinates as vairbales from an xarray dataset |
| CachingIndex          | `indexes.CachingIndex` | Cache data recieved at this step |
| TemporalIndex         | `indexes.TemporalIndex` | Grab temporal samples around requested time |

## Interfaces

All paths shown below are relative to `edit.pipeline`.

| Name | Path | Purpose |
| ---- | ---- | ------: |
| NormaliseInterface    | `interfaces.NormaliseInterface`    | Setup normalisation of data indexes |

## Iterators

All paths shown below are relative to `edit.pipeline`.

| Name | Path | Purpose |
| ---- | ---- | ------: |
| Iterator    | `iterators.Iterator`    | Base Iterator in time sequence |
| RandomIterator    | `iterators.RandomIterator`    | Random time indexing |
| CombineDataIterator    | `iterators.CombineDataIterator`    | Combine pipelines, alternating between |
| FakeData    | `iterators.FakeData`    | Iterate through fake data, for testing |
