import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import tarfile
    import gzip
    import shutil
    from pathlib import Path
    import numpy as np
    from datetime import datetime
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import marimo as mo
    import os

    from dask.distributed import Client
    import xarray as xr
    return Path, datetime, os, xr


@app.cell
def _(Path):
    UNPACKED_DIR = Path.home() / 'hadisd' / 'unpacked'       # We need a place on disk to unpack the archives
    PROCESSING_DIR = Path.home() / 'hadisd' / 'processing'   # We need to cache some data on disk during reprocessing
    return PROCESSING_DIR, UNPACKED_DIR


@app.cell
def _(UNPACKED_DIR):
    files = list(UNPACKED_DIR.glob('*.nc'))
    len(files)
    return (files,)


@app.cell
def _(xr):

    def simplify(ds):
        lats = xr.DataArray(data=[ds.latitude.values[0]] * len(ds.time), coords={'time': ds.time})
        lons = xr.DataArray(data=[ds.longitude.values[0]] * len(ds.time), coords={'time': ds.time})
        elev = xr.DataArray(data=[ds.elevation.values[0]] * len(ds.time), coords={'time': ds.time})
        ds = ds.reset_coords(names=('latitude', 'longitude', 'elevation'), drop=True)
        ds['lat'] = lats
        ds['lon'] = lons
        ds['elev'] = elev

        ds = ds.drop_attrs()
        return ds
    return (simplify,)


@app.cell
def _(files):
    filegroups = [files[i:i + 10] for i in range(0, len(files), 10)]
    print(len(filegroups))  # We come up with 134 such file groupings from the test data
    return (filegroups,)


@app.cell
def _():
    decades = [('1800', '1930'), # Just in case there is undocumented early data
               ('1930', '1940'), ('1940', '1950'), # Dataset begins in 1930, start by decade here
               ('1950', '1960'), ('1960', '1970'), ('1970', '1980'), 
               ('1980', '1990'), ('1990', '2000'), ('2000', '2010'), # 1980 is a common time to start from
               ('2010', '2020'), ('2020', '2030')
              ]
    return (decades,)


@app.cell
def _(PROCESSING_DIR, datetime, decades, filegroups, os, simplify, xr):
    # This takes around 20-30 seconds per grouping. If you just want to get the hang of it, limit it to three groupings
    # Otherwise, the test set have 67 groupings, so will take around half an hour to run
    # The full set of stations will take several hours.

    # For testing, just try three file groups

    # for i, fg in enumerate(filegroups[3]):  # Use me to test three file groupings
    for i, fg in enumerate(filegroups):       # Use me to process all downloaded data
        print(f"Processing group {i} of {len(filegroups)}")
        print(datetime.now().time())
        loaded = [xr.open_dataset(f, engine='h5netcdf') for f in fg]
        simplified = [simplify(_ds) for _ds in loaded]
        merged = xr.concat(simplified, dim='report')

        for d in decades:
            decadal = merged.sel(time=slice(*d))
            if len(decadal.time):
                filename = PROCESSING_DIR / f'{d[0]}-{d[1]}-sg{i}.nc'
                if not os.path.exists(filename):
                    decadal.to_netcdf(filename)
                else:
                    print(f"{filename} exists, skipping")
    return


@app.cell
def _():

    print('done')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
