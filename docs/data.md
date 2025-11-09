# PyEarthTools and Data Access

Data is not provided by PyEarthTools (either directly or as a cloud service), it must be downloaded by the user, and any data licenses must be observed by the user.

- [Using PyEarthTools in an HPC Environment](#Using-PyEarthTools-in-an-HPC-Environment)
  - [Accessing Data with PyEarthTools at NCI](#Accessing-Data-with-PyEarthTools-at-NCI)
  - [Connecting PyEarthTools to a new dataset in any HPC facility by coding a new data accessor](#connecting-pyearthtools-to-a-new-dataset-in-any-hpc-facility-by-coding-a-new-data-accessor)
- [Using PyEarthTools on a Workstation or Laptop](#Using-PyEarthTools-on-a-Workstation-or-Laptop)
  - [With pre-configured datasets, supported by PyEarthTools, which you download from the internet](#with-pre-configured-datasets-supported-by-pyearthtools-which-you-download-from-the-internet)
  - [Connecting PyEarthTools to a new dataset on a workstation or laptop](#connecting-pyearthtools-to-a-new-dataset-on-a-workstation-or-laptop)

## Using PyEarthTools in an HPC Environment

PyEarthTools can efficiently access large, multi-terabyte data sets. These data sets are typically held on-disk at dedicated computing facilities.

At the moment, PyEarthTools has existing integrations with the data holdings at three HPC facilities: 
- NCI (Australia). 
- Met Office (UK).
- Earth Sciences New Zealand (formerly NIWA). 

If you are working at another HPC facility, feel free to get in touch to discuss how to most effectively utilise PyEarthTools in your environment.

### Accessing Data with PyEarthTools at NCI

The package `site_archive_nci` (which is present in some of the tutorials) is the NCI data accessor. `site_archive_nci` provides access to key Earth system datasets. Currently the following datasets are supported:

| Name        | Description |
| ---         |   --------- |
| ERA5        | ECWMF ReAnalysis v5                                     |
| ACCESS      | Australian Community Climate and Earth-System Simulator |
| AGCD        | Australian Gridded Climate Data                         |
| BRAN        | Bluelink ReANalysis                                     |
| OceanMaps   | Ocean Modelling and Analysis Prediction System          |
| MODIS       | MODerate resolution Imaging Spectroradiometer           |
| Himawari    | Himawari 8/9 satellite data                             |
| Rainfields3 | Rainfields3 Australia-wide radar mosiac 2km^2 (Ausm310) |
| BARRA       | Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia    |
| BARPA       | Bureau of meteorology Atmospheric Regional Projections for Australia                   |
| BARRA_V2    | Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia v2 |

Additionally, you can explore the [geonetwork](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/home) to find additional data sets of interest which you might like to utilise by writing a custom PyEarthTools data accessor.

### Connecting PyEarthTools to a new dataset in any HPC facility by coding a new data accessor

For on-disk data access, you will need to create a new accessor based on the FileSystemArchive class. Intstructions for this still need to be written. In the meantime, refer to the [NCI site archive source code](https://github.com/ACCESS-Community-Hub/PyEarthTools/tree/develop/packages/nci_site_archive) for examples. 

The [HadISD tutorials](project:./notebooks/Gallery.md#Working-with-Station-Data-(Medium-Hardware-Requirements)) also demonstrate the process of creating a new data accessor. While these tutorials focus on connecting to the HadISD dataset, the patterns in these tutorials are repeatable and can be used for other datasets.

Additional considerations and rules-of-thumb for HPC environments are:

- Use large files rather than many small files. This makes formats like GRIB and NetCDF more appropriate than Zarr in many cases.
- If using dask for chunking data, use largeish chunks, aligned to the time dimension (or primary index dimension).
- Do not zip up large datasets. Use internal zip compression. Zip is inherently single-threaded, so can require a long, slow, bottlenecked decompression step before data subsets can be read from a large file. 
- Use a format like Parquet for point clouds, station data or other irregularly-spaced, sparse data.

## Using PyEarthTools on a Workstation or Laptop

You can use PyEarthTools successfully on a workstation or laptop with data you download yourself. 

While many geoscience datasets are so large (e.g. hundreds of terabytes) that they can only be used effectively in HPC environments, there are also many smaller datasets of interest which can be downloaded on a workstation or laptop.

### With pre-configured datasets, supported by PyEarthTools, which you download from the internet

The [Quick Start](project:./notebooks/Gallery.md#Quick-Start-(Low-Hardware-Requirements)) tutorials can run on a 4GB GPU, and include the download step for fetching around 3-10GB of data. They will also work in HPC environments.

The [station data](project:./notebooks/Gallery.md#Working-with-Station-Data-(Medium-Hardware-Requirements)) tutorials do not need a GPU, but require more data. They have been tested on a laptop with 36GB of RAM and as well as an HPC node with over 100GB of RAM. 29GB of station data will be downloaded. Additional disk space is needed for reprocessing the data, although intermediate files can later be deleted. These notebooks may require user modification to run with less than 36GB of RAM but it should be possible with at least 16G of RAM. 

### Connecting PyEarthTools to a new dataset on a workstation or laptop

For on-disk data access, you will need to create a new accessor based on the `pyearthtools.data.ArchiveIndex` class (or `pyearthtools.data.Index` for some use cases). Instructions for this still need to be written. In the meantime, refer to the [NCI site archive source code](https://github.com/ACCESS-Community-Hub/PyEarthTools/tree/develop/packages/nci_site_archive) for examples. 

The [HadISD tutorials](project:./notebooks/Gallery.md#Working-with-Station-Data-(Medium-Hardware-Requirements)) also demonstrate the process of creating a new data accessor. While these tutorials focus on connecting to the HadISD dataset, the patterns in these tutorials are repeatable and can be used for other datasets.

Additional considerations and rules-of-thumb for working on workstations or laptops are:

- Using a storage format like zarr is suitable, because they can efficiently use small files to index data
- RAM is likely to be more constrained, so consider limiting the number of workers for tools like dask or PyTorch
- Some datasets have so-called ARCO (analysis-ready, cloud-optimised) versions available. Downloading an entire dataset in this fashion may be cost-prohibitive and inefficient for model training, but may be suitable for occasional access or to download model initial conditions for a single model run. 
- There are also some versions of some datasets which have been heavily compressed using lossy compression techniques, but are still close analogs for the original data. These could be used for model training, but there will be caveats as to the accuracy of such models due to the lossy compression of the training data.
