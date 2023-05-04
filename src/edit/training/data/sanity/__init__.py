"""
Sanity Checking tools for a Data Pipeline.

Used to ensure quality of data, and diagnose any issues with loading or filtering incoming data.

| Tools | Description |
| ----- | ----------- |
| summary | Print out summary of shape and type of each [DataStep][edit.training.data.templates.DataStep] |
| plot    | Plot samples from each [DataStep][edit.training.data.templates.DataStep] either using `__getitem__` or `__iter__` |
"""

from edit.training.data.sanity import iterator_retrieval
from edit.training.data.sanity.summary import summary
from edit.training.data.sanity.plotting import plot
