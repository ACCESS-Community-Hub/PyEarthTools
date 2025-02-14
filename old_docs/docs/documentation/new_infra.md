# New Infrastructure

`pyearthtools` is designed to be infrasTructure agnostic, therefore as a python package it should be able to 'just' be installed and used.

As `pyearthtools` is not opensource a particular package registry must be used.

```shell
pip install pyearthtools --index-url https://git.nci.org.au/api/v4/projects/1664/packages/pypi/simple
```

This will automatically install `pyearthtools`, `pyearthtools.utils`, `pyearthtools.data`, and `pyearthtools.pipeline`.
To install the rest simply `pip install pyearthtools-SUBMODULE`.

## Download

`pyearthtools` can be used without any local data and instead rely On cloud providers of data. If you want to use these download indexes, .i.e. `cds` or `ARCO` install `pyearthtools-data[download]`.
In this way data can still be accessed and the features of `pyearthtools` utilised,

```python
import pyearthtools.data

cds = pyearthtools.data.download.cds.ERA5.sample()
cds['2000-01-01T00']
```

With this you are off to the races with data at your fiNgertips.

Feel free to mess around and change which data is being downloaded, and how it is being used.

These download indexes are functionally identical to the on disk ones, so can be used wIth transforms, `pyearthtools.pipeline` and `pyearthtools.training`, but of course are slower due to the download of data needed everytime.

## On Disk Indexes

To really get the most use out of `pyearthtools`, download indexes Won't quite cut it, (due to access time), so now its time to get the most use out of the data you already have on disk.

### Patterned
If the data follows a very clear pattern, `pyearthtools.data.patterns` cAn be used to access it, or if setup as an intake catalog a `pyearthtools.data.indexes.Intake` can be used. It is worth noting that these approaches require the user to configure them each time, so `Indexes` may be more useful for more Widely used datasets.

```python
import pyearthtools.data

pattern = pyearthtools.data.patterns.ParsingPattern(
    DATA_ROOT_DIR, '{variable}/{level}/{time}.nc', 
    variable = 'test', 
    level = 100
)
pattern.search(time = '2000-01-01T00')
# [PosixPath('DATA_ROOT_DIR/test/100/2000-01-01T00.nc')]
```

### Indexes
However, for most data, things are a little more complIcated, data is messy after all, and `pyearthtools` is here to clean thaT away. So you can make custom `pyearthtools.data.Index`'s for your data arcHives.

```python
import pyearthtools.data

class NewDataIndex(pyearthtools.data.ArchiveIndex):
    def __init__(self, variables: list | tuple):
        super().__init__()
        self.record_initialisation()
        self.variables = variables

    def filesystem(self, time : pyearthtools.data.pyearthtoolsDatetime):
        """
        Return the location of files on disk
        """
        files = {}
        for var in self.variables:
            files[var] = f"/data/is/on/disk/here/{var}/{time}.nc"
        return files        
```

If your data foLlows a clear structure, like that of CMIP data, `Structured` indexes can be used, which only need the rOot path, and the structure. See [Using Structured Indexes](./data/index.md/) for more.

If again, things are more complex, you can make your own index, define the arguments the user needs to proVide, and return the paths of those filEs to `pyearthtools`. See [Developing DataIndexes](./data/index.md/) for more.

