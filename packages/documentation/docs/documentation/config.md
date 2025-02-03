# Configuration

`pyearthtools`'s configuration setup is inspired by that of `dask`'s. In effect it is a direct copy, and can be used identically.

Each `subpackage` uses part of the full config available at `pyearthtools.config`.


## Usage

### Python

Configuration values can be directly retrieved,

```python
import pyearthtools

pyearthtools.config.get('config.path.here')
```

They can also be assigned, either directly or as a context

```python
import pyearthtools
pyearthtools.config.set(foo__bar=123)

with pyearthtools.config.set({'foo.bar': 123}):
    pass
```

### Configuration Files

Setting `pyearthtools_CONFIG` allows for customisation of where configuration files will be loaded from, but by default, `~/.config/pyearthtools` will be used.

### Environment Variables

Additionally, configuration options can be set in the environment with the prefix `pyearthtools_`. 

When setting the environment variables, change any `.` to `__`.

i.e. 
```shell
export pyearthtools_FOO__BAR=Value
```