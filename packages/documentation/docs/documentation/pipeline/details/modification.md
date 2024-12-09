# Modification

Modifications are way to add advanced functionality to a pipeline which can directly modify the index of data being retrieved, and have full access to parent pipeline above it.

Modifications which are implemented already, include index modifiers and a caching step. 

Particularly, these steps subclass from `pyearthtools.pipeline.PipelineIndex`, and are expected to provide an implementation of `__getitem__`. Unlike an operation, this is not applied, it is accessed instead of using the root index. 

Additionally, `undo_func` can be implemented to reverse the effect of the `PipelineIndex` when calling undo on the pipeline.

When used within a `Pipeline`, `set_parent_record` is called to record what steps exist above this `PipelineIndex`. Using this record, `parent_pipeline`, and `as_pipeline` can be called to find the parent pipeline and the complete pipeline to this step respectively can be accessed.

## Example

Below is the implementation of `IdxOverride` to override the index whenever the pipeline is accessed,

```python
class IdxOverride(PipelineIndex):
    """Override `idx` on any `__getitem__` call"""

    def __init__(self, index: Any):
        super().__init__()
        self.record_initialisation()

        self._index = index

    def __getitem__(self, *_, **__):
        # Instead of using the pipelines indexes, use this which overrides
        # the data flow
        return self.parent_pipeline()[self._index]
```

Once initialised, this can be used in a pipeline like any other step. 


```python
pyearthtools.pipeline.Pipeline(
    pyearthtools.data.archive.ERA5.sample(),
    pyearthtools.pipeline.modifications.IdxOverride('2000-01-01T00')
)

# All indexes to this pipeline will return data for '2000-01-01T00'
```

