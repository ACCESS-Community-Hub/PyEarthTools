"""
Collection of [DataIterators][edit.training.data.DataIterator] for use in Data Pipelines

All implement [DataIterator][edit.training.data.DataIterator], so will provide date based iterations and error catching

| Name                | Description |
| ------------------- | ----------- |
| [Iterator][edit.training.data.iterators.iterator]            | Basic Iterator  |
| [Iterator][edit.training.data.iterators.random]            | Iterator which Randomly Samples dates  |
| [CombineDataIterator][edit.training.data.iterators.combine]    | Combine Multiple DataIterators together and alternate between samples |
| [FakeData][edit.training.data.iterators.fakedata]            | Fake Data loading process to eliminate data loading times |
"""

from edit.training.data.iterators.iterator import Iterator
from edit.training.data.iterators.random import RandomIterator
from edit.training.data.iterators.combine import CombineDataIterator
from edit.training.data.iterators.fakedata import FakeData

## Backwards compatible api
from edit.training.data.indexes.temporal import TemporalIndex as TemporalInterface
from edit.training.data.indexes.temporal import TemporalIndex as TemporalIterator
