# Filter

As data can contain superious artefacts, `pyearthtools.pipeline` adds the abilty to filter data, checking against a condition and raising an exception if 'bad' data is encountered.

When iterating over a pipeline, these filter errors are automatically skipped and the next sample retrieved.

For direct indexing, the exception is raised, but can be caught by downstream processes.

A `Filter` is another `PipelineStep` so can be directly integrated into a `Pipeline`.

As it subclasses from `PipelineStep` it is capable of auto splitting tuples.

## Example
Below is the implementation of the `TypeFilter` bundled with `pyearthtools.pipeline`

```python
class TypeFilter(Filter):
    """
    Filter if type is wrong
    """

    def __init__(self, valid_types: Union[tuple[Type], Type], *, split_tuples: bool = False):
        super().__init__(split_tuples=split_tuples)
        self.record_initialisation()

        if not isinstance(valid_types, tuple):
            valid_types = (valid_types,)
        self._valid_types = valid_types

    def filter(self, sample) -> None:
        if not isinstance(sample, self._valid_types):
            raise PipelineFilterException(sample, f"Expecting type/s {self._valid_types}")
```

As it subclasses from `Filter` it is expected to implement a `filter` function, raising a `PipelineFilterException` if invalid data is encountered.

Additionally, a filter can subclass from `FilterCheck` and implement `check` returning a boolean, with the exception raising automated.