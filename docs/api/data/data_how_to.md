# Data API How-To Guide

The PyEarthTools [Data API](/api/data/data_index.md) provides Data Accessors, which load data from disk or over the network and into an Xarray data format for further processing.

They handle the nuances of how the data set is stored and organised, such as how to walk the filesystem, how to match a user query to the files on disk, and how to subset the requested variables out of the data structure. They may also handle any transformations which are needed to the raw data, such as file compression.

A more detailed how-to guide will be written in future describing how to use the various classes in the data module. 

For a general overview and examples of how to make use some of the data module's functionality, see:

- [data-specific tutorials](project:/notebooks/Gallery.md#Deep-Dive---The-Data-Module) for tutorials showing how to work with data module classes.
- [PyEarthTools and data access](/data.md) for an overview of how to fetch data and load it for use
- [HadISD Tutorial Seven - Parquet Accessor](project:/notebooks/stationdata/HadISD-Seven-ParquetAccessor.ipynb) for a tutorial showing how to create a data accessor for weather station data
