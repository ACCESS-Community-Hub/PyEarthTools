# Configuration

`edit`'s configuration setup is inspired by that of `dask`'s. In effect it is a direct copy, and can be used identically.

Each `subpackage` uses part of the full config available at `edit.config`.


## Usage

### Python

Configuration values can be directly retrieved,

```python
import edit

edit.config.get('config.path.here')
```

They can also be assigned, either directly or as a context

```python
import edit
edit.config.set(foo__bar=123)

with edit.config.set({'foo.bar': 123}):
    pass
```

### Configuration Files

Setting `EDIT_CONFIG` allows for customisation of where configuration files will be loaded from, but by default, `~/.config/edit` will be used.

### Environment Variables

Additionally, configuration options can be set in the environment with the prefix `EDIT_`. 

When setting the environment variables, change any `.` to `__`.

i.e. 
```shell
export EDIT_FOO__BAR=Value
```