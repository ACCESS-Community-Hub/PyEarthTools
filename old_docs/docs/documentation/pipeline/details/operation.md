# Operations

An `pyearthtools.pipeline.Operation` is the core class which makes up the steps of a pipeline. All `operations.*` subclass from it. 

Additonally, with `PipelineStep` it provides a number of configuration options to help data flow, splitting of tuples, and raise/warn an exception on invalid types.

`Operation` provides the interface in which functions are applied on a forward pass of the pipeline, and the functions to run on an undo operation. 

## Functions

### Record initialisation

To record the init args / kwargs call `self.record_initialisation()`.

### `__init__`
```
Base `Pipeline` Operation,

Allows for tuple spliting, and type checking

Args:
    split_tuples (Literal['apply', 'undo', True, False], optional):
        Split tuples on associated actions, if bool, apply to all functions. Defaults to False.
    recursively_split_tuples (bool, optional):
        Recursively split tuples. Defaults to False.
    operation (Literal['apply', 'undo', 'both'], optional):
        Which functions to apply operation to. 
        If not 'apply' apply does nothing, same for `undo`. Defaults to "both".
    recognised_types (Optional[Union[tuple[Type, ...], Type, dict[str, Union[tuple[Type, ...], Type]]] ], optional):
        Types recognised, can be dictionary to reference different types per function Defaults to None.
    response_on_type (Literal['warn', 'exception', 'ignore', 'filter'], optional):
        Response when invalid type found. Defaults to "exception".
```

`operation` controls if the functions are actually used. If `operation = 'undo'`, when calling `apply` on the operation, the sample will be returned as is. The same goes for `operation = 'apply'` but for `undo`.

### apply

Runs `apply_func` upon the sample, splitting tuples if configured. 

The user must provide `apply_func` in the subclass if operation is either `apply` or `both`.

### undo

Runs `undo_func` upon the sample, splitting tuples if configured. 

The user must provide `undo_func` in the subclass if operation is either `undo` or `both`.

### T

Creates a transposed operation, swapping `apply` and `undo`. 

Useful for reverse pipelines for when an inverse operation is not provided for an operation.

```python
example_operation = Operation(...)

inverse = example_operation.T
```

## Example Implementation

Here is the implementation of `numpy.reshape.Squish`, to flatten a one element axis of an array

```python
from typing import Union, Optional, Any
import numpy as np

from pyearthtools.pipeline.operation import Operation

class Squish(Operation):
    """
    Operation to Squish one Dimensional axis at 'axis' location
    """

    _override_interface = ["Delayed", "Serial"] # Which parallel interfaces to use in order of priority.
    _interface_kwargs = {"Delayed": {"name": "Squish"}}

    def __init__(self, axis: Union[tuple[int, ...], int]) -> None:
        """Squish Dimension of Data

        Args:
            axis (Union[tuple[int, ...], int]):
                Axis to squish at
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(np.ndarray),
        )
        self.record_initialisation() # Record init kwargs for saving
        self.axis = axis

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        return np.squeeze(sample, self.axis)

    def undo_func(self, sample: np.ndarray) -> np.ndarray:
        return np.expand_dims(sample, self.axis)
```

For a user to add their own the `Operation`, provide `apply_func` and/or `undo_func`. 

If tuples are to be split, set `split_tuples=True`, and `operation` to the operations to actually apply.