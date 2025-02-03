# Adding your Index

If a user has created a new index, it is possible to register it at `pyearthtools.data.archive`.

```python
@pyearthtools.data.archive.register_archive("NewData")
class NewData:
    def __init__(self, initialisation_args):
        pass
```

This new index would now be available at `pyearthtools.data.archive.NewData` once the package containing this index is imported.

```python
import pyearthtools.data

pyearthtools.data.archive.NewData(initialisation_args)

```
