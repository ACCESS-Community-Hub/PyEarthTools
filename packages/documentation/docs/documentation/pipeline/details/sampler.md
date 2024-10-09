# Samplers

Samplers control how data is returned / sampled when the pipeline is being iterated over. 

When directly indexing into the `Pipeline` no changes are applied.

## Example

```python
import edit.pipeline

iterator = edit.pipeline.iterators.DateRange('2000', '2005', '6 hours')
sampler  = edit.pipeline.samplers.Random(buffer_len = 10)

pipeline = edit.pipeline.Pipeline.sample(iterator = iterator, sampler = sampler)

iter(pipeline)
```

Using the `DateRange` iterator will cause samples to be pulled from the indexes at the times within the range in order. Then with use of the `sampler`, samples will be provided to the `sampler` which builds a buffer, and once at size, begins to randomally sample from it.

## Provided

With `edit.pipeline` multiple samplers by default are provided.

| Name | Description |
| ---- | ----------- |
| `Default` | Default sampler, does nothing |
| `SuperSampler` | Combine multiple samplers, using one after another |
| `Random` | Randomally sample |
| `DropOut` | Drop out samples at set interval |
| `RandomDropOut` | Drop out samples randomally |


## Implementation

Below is the implementation of the `Default` sampler.

```python
from typing import Any, Generator, Union

from edit.pipeline.recording import PipelineRecordingMixin

class Default(Sampler):
    """
    Default Sampler

    Simply passes back any object given to it.
    """

    def generator(self) -> Generator[Any, Any, Any]:
        obj = EmptyObject()  # Yields an Empty Object to start with

        # Run forever until None is encountered
        while True:
            # Yield the prior object and capture what is sent
            obj = yield obj
            # If None is encountered, exit the generator
            if obj is None:
                break
```

The `Sampler`'s use the `generator` function to create a `generator` to manage the sampling. 

## Super Sampler

Adding two samplers together will create a `SuperSampler` which will nest the samplers together,

```python
Random(10) + DropOut(4)
# SuperSampler
```


## Creating your Own

If the `Sampler` is not ready to yield data, provide an `EmptyObject()`, which informs the controller to keep iterating. To get samples from the main iteration loop capture result of the `yield` call, and then when ready to give back data, yield it.

And then when done, break out, which will end the sampling.
