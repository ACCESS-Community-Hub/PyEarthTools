# Adding your Index

If a user has created a new index, it is possible to register it at `edit.data.archive`.

```python
@edit.data.archive.register_archive("NewData")
class NewData:
    def __init__(self, initialisation_args):
        pass
```

This new index would now be available at `edit.data.archive.NewData` once the package containing this index is imported.

```python
import edit.data

edit.data.archive.NewData(initialisation_args)

```
