# Overview

`pyearthtools` aims to reduce the time and effort required to rapidly iterate with and work on Environmental Data. Within `pyearthtools.training`, this data can be easily loaded and prepared for Machine Learning Training.

## Core Data Structures

### `pyearthtools.data`

`pyearthtools.data` creates and uses DataIndexes, which define the API to access various forms of data.

Some allow temporal indexing, while others allow file system based operations.

<!-- - [`DataIndex`][pyearthtools.data.DataIndex] is the base implementation of data sources whcih can be accessed with a single timestep.
- [`OperatorIndex`][pyearthtools.data.OperatorIndex] builds upon the [`DataIndex`][pyearthtools.data.DataIndex] to provide methods in which to retrieve a sequence of data. This object allows for date resolution to infer the data retrieval scope. -->

The power of these objects come from obfuscating the file structures, patterns and oddities associated with each dataset, and thus the code to or index required to retrieve data after intialised becoming identical. See below for an example...

```python
import pyearthtools.data

##Common Date of interest
doi = '2022-02-12T000'

##Access ERA5
ERA5 = pyearthtools.data.archive.ERA5(variable = '2t', level = 'single')
##Access BRAN
BRAN = pyearthtools.data.archive.BRAN(variable = 'ocean_temp', type = 'daily')

## Retrieve Data
ERA5(doi) # Get ERA5 data at doi
BRAN(doi) # Get BRAN data at doi

```

### `pyearthtools.training`

`pyearthtools.training` allows a user to specify a data pipeline with which to prepare data for a Machine Learning model training and operation,

## Issues

To report issues or bugs please use the Issues on the [GitLab](https://git.nci.org.au/bom/dset/pyearthtools-package)

When using this plrease give as much detail as possible, including but not limited to:

- Example Code to reproduce the error
- Expected Behaviour
- Urgency
