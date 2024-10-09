# Iterator

As explained in [Using](./../using.md), pipelines can be iterated over, and use `Iterator`'s to define the indexes in which to retrieve data. 

These objects subclass from `edit.pipeline.Iterator`, and must simply implement a `__iter__` function which is expected to return / yield indexes. These are then used in the pipelines `__iter__` to `__getitem__` data.

## Example

Below is the implementation for the `Range` iterator,

It uses the python range function as the iterator.

```python
class Range(Iterator):
    """
    Range based Iterator

    Constructs a `range` object and yields all elements within.
    """

    def __init__(self, min: int, max: int, step: int = 1):
        """
        Construct Range Iterator

        Args:
            min (int):
                Minimum value of range
            max (int):
                Maximum value of range
            step (int, optional):
                Step of range. Defaults to 1.
        """
        super().__init__()
        self.record_initialisation()

        self._range = tuple(range(min, max, step))

    def __iter__(self) -> Generator[Hashable, None, None]:
        for i in self._range:
            yield i
```

## Other functions

`Iterators` can be added together with `+`, which will produce a `SuperIterator` consuming one iterator after another.

```python
Range(0, 10, 2) + Range(20, 50, 1)
# SuperIterator
```

Addtionally, a `randomise` function is available on each iterator to wrap the iterator in question in a randomiser

```python
Range(0,10,1).randomise()
# Random Iterator over Range
```

All samples from an iterator can be accessed by calling `samples`.

```python
Range(0,10,1).samples
#(0,1,2,3,4,5,6,7,8,9)
```