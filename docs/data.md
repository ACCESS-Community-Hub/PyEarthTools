# PyEarthTools and Data Access

Data is not provided by PyEarthTools (either directly or as a cloud service), it must be downloaded by the user, and any data licenses must be observed by the user.

- [Using PyEarthTools in an HPC Environment](#Using-PyEarthTools-in-an-HPC-Environment)
- [Using PyEarthTools on a Workstation or Laptop](#Using-PyEarthTools-on-a-Workstation-or-Laptop)

## Using PyEarthTools in an HPC Environment

PyEarthTools can efficiently access large, multi-terabyte data sets. These data sets are typically held on-disk at dedicated computing facilities.

At the moment, PyEarthTools has existing integrations with the data holdings at three HPC facilities: 
- NCI (Australia). 
- Met Office (UK).
- Earth Sciences New Zealand (formerly NIWA). 

If you are working at another HPC facility, feel free to get in touch to discuss how to most effectively utilise PyEarthTools in your environment.

The package `site_archive_nci` (which is present in some of the tutorials) is the NCI data accessor. `site_archive_nci` provides access to key Earth system datasets.

Additionally, you can explore the [geonetwork](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/home) to find additional data sets of interest which you might like to utilise by writing a custom PyEarthTools data accessor.

## Using PyEarthTools on a Workstation or Laptop

You can use PyEarthTools successfully on a workstation or laptop with data you download yourself. 

While many geoscience datasets are so large (e.g. hundreds of terabytes) that they can only be used effectively in HPC environments, there are also many smaller datasets of interest which can be downloaded on a workstation or laptop.

The [Quick Start](project:./notebooks/Gallery.md#Quick-Start-(Low-Hardware-Requirements)) tutorials can run on a 4GB GPU, and include the download step for fetching around 3-10GB of data. They will also work in HPC environments.

The [station data](project:./notebooks/Gallery.md#Working-with-Station-Data-(Medium-Hardware-Requirements)) tutorials do not need a GPU, but require more data. They have been tested on a laptop with 36GB of RAM and as well as an HPC node with over 100GB of RAM. 29GB of station data will be downloaded. Additional disk space is needed for reprocessing the data, although intermediate files can later be deleted. These notebooks may require user modification to run with less than 36GB of RAM but it should be possible with at least 16G of RAM.

Some datasets have so-called ARCO (analysis-ready, cloud-optimised) versions available. Downloading an entire dataset in this fashion is still cost-prohibitive and inefficient for model training, and this approach is intended more for occasional access or to download model initial conditions at a single time. 

There are also some versions of some datasets which have been heavily compressed using lossy compression techniques, but are still close analogs for the original data. These could be used for model training, but there will be caveats as to the accuracy of such models due to the lossy compression of the training data.
