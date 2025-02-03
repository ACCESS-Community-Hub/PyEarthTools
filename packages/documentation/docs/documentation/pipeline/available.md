# Available Operations

Within `pyearthtools.pipeline.operations` are the steps available to make up a pipeline. 

To add your own, see [Making your own Operation](details/operation.md). Essentially subclass `pyearthtools.pipeline.Operation`.

There are four main operation types available
- xarray
- numpy 
- dask
- Transforms

## xarray

As `pipeline` utilises `data` to provide the base indexes in which to actually get data from, a very common data type will be `xarray` objects, either `Dataset` or `DataArrays`.
Therefore, `xarray` operations are common place. 

Here are the default operations included with `pyearthtools.pipeline`, accessible at `pyearthtools.pipeline.operations.xarray.{Category}.{Name}`.

| Category | Description | Available |
| -------- | ----------- | --------- |
| Compute  | Call compute on an xarray object | `Compute` |
| Chunk  | Rechunk xarray object | `Chunk` |
| conversion | Convert datasets between numpy or dask arrays | `ToNumpy`, `ToDask` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Join tuples of xarray objects | `Merge`, `Concatenate` |
| metadata | Modify or keep metadata | `Rename`, `Encoding`, `MaintainEncoding`, `Attributes`, `MaintainAttributes` |
| normalisation | Normalise datasets | `Anomaly`, `Deviation`, `Division`, `Evaluated` |
| reshape | Reshape datasets | `Dimension`, `CoordinateFlatten` |
| select | Select elements from dataset's | `SelectDataset`, `DropDataset`, `SliceDataset` |
| sort | Sort variables of a dataset | `Sort` |
| split | Split datasets | `OnVariables`, `OnCoordinate` |
| values | Modify values of datasets | `FillNan`, `MaskValue`, `ForceNormalised`, `Derive` |
| remapping | Reproject data | `HEALPix` | 

## numpy

Typically for Machine learning training, data must be converted to arrays and then tensors. So using `xarray.conversion.ToNumpy` data can be turned into arrays. Once in that form, operations can still be applied.

Here are the default operations included with `pyearthtools.pipeline`, accessible at `pyearthtools.pipeline.operations.numpy.{Category}.{Name}`.


| Category | Description | Available |
| -------- | ----------- | --------- |
| augument | Augument numpy data | `Rotate`, `Flip`, `Transform` | 
| conversion | Convert between data types | `ToXarray`, `ToDask` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Combine tuples of `np.ndarrays` | `Stack`, `VStack`, `HStack`, `Concatenate` |
| normalisation | Normalise arrays | `Anomaly`, `Deviation`, `Division`, `Evaluated`  |
| reshape | Reshape numpy array | `Rearrange`, `Squish`, `Expand`, `Flatten`, `SwapAxis` |
| select | Select elements from array | `Select`, `Slice` |
| split  | Split numpy arrays into tuples | `OnAxis`, `OnSlice`, `VSplit`, `HSplit` |
| values | Modify values of arrays | `FillNan`, `MaskValue`, `ForceNormalised` |

## dask

For optimisation and rate increases, data can be kept in dask arrays for as long as possible. This section directly mimics the `numpy` one but instead operates on dask arrays.

Here are the default operations included with `pyearthtools.pipeline`, accessible at `pyearthtools.pipeline.operations.dask.{Category}.{Name}`.


| Category | Description | Available |
| -------- | ----------- | --------- |
| augument | Augument numpy data | `Rotate`, `Flip`, `Transform` | 
| Compute  | Call compute on an dask object | `Compute` |
| conversion | Convert between data types | `ToXarray`, `ToNumpy` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Combine tuples of `np.ndarrays` | `Stack`, `VStack`, `HStack`, `Concatenate` |
| normalisation | Normalise arrays | `Anomaly`, `Deviation`, `Division`, `Evaluated`  |
| reshape | Reshape numpy array | `Rearrange`, `Squish`, `Expand`, `Flatten`, `SwapAxis` |
| select | Select elements from array | `Select`, `Slice` |
| split  | Split numpy arrays into tuples | `OnAxis`, `OnSlice`, `VSplit`, `HSplit` |
| values | Modify values of arrays | `FillNan`, `MaskValue`, `ForceNormalised` |

## Transforms

In addition to the operations that can be applied, `pyearthtools.data.transforms` can also be applied. They can be directly be included within the `Pipeline` and will only be called on the `apply` step.

If `transforms` need to be applied on the `undo` step or with other specifications `pyearthtools.pipeline.operations.Transforms` can be used. 

```python
pyearthtools.pipeline.operations.Transforms(
    transforms = pyearthtools.data.transforms.region.Lookup('Oceania'), # Applied on both apply and undo
    apply = pyearthtools.data.transforms.coordinates.Expand(dim = [0]), # Only on apply
    undo = pyearthtools.data.transforms.coordinates.Drop(dim = [0]), # Only on undo

)
```


## Modifications

Additionally to the simple operations, `modifications` are available, these are named so as they modify the flow of data and indexes to a greater level.

Here are the default modifications included with `pyearthtools.pipeline`, accessible at `pyearthtools.pipeline.modifications.{Name}`.

| Category | Description | Available |
| -------- | ----------- | --------- |
| Cache | Cache data to disk at that point in the pipeline | `Cache`, `StaticCache` |
| Idx_Modification | Modify the index being retrieved | `IdxModifier`, `IdxOverride`, `TimeIdxModifier`, `SequenceRetrieval`, `TemporalRetrieval`|