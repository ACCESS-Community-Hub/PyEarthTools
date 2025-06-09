# Data API How-To Guide

The PyEarthTools Data API provides Data Accessors, which load data from disk or over the network and into an Xarray data format for further processing.

They handle the nuances of how the data set is stored and organised, such as how to walk the filesystem, how to match a user query to the files on disk, and how to subset the requested variables out of the data structure. They may also handle any transformations which are needed to the raw data, such as file compression.

A more detailed how-to guide will be written in future. For now see the [data-specific tutorials](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html#PyEarthTools---Data-Module) in the gallery.
