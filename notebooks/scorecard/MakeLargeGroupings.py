import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    # https://www.metoffice.gov.uk/hadobs/hadisd/v343_2025f/index.html
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    from datetime import datetime
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import marimo as mo

    from dask.distributed import Client
    import xarray as xr
    return Path, xr


@app.cell
def _(Path):
    # A spot to put the data on disk. We keep both the data as-downloaded and the reprocessed version, so you might need up to 50GB free in order to make this work.

    PROCESSING_DIR = Path.home() / 'hadisd' / 'processing'   # We need to cache some data on disk during reprocessing
    DECADAL_DIR = Path.home() / 'hadisd' / 'by_decade'    # This will hold the final form of our data
    return DECADAL_DIR, PROCESSING_DIR


@app.cell
def _():
    decades = {
        'early': ('1800', '1930'), # Just in case there is undocumented early data
        '1930': ('1930', '1940'),  # Dataset begins in 1930, start by decade here 
        '1940': ('1940', '1950'),
        '1950': ('1950', '1960'), 
        '1960': ('1960', '1970'), 
        '1970': ('1970', '1980'), 
        '1980': ('1980', '1990'), 
        '1990': ('1990', '2000'), 
        '2000': ('2000', '2010'), 
        '2010': ('2010', '2020'), 
        '2020': ('2020', '2030')
    }
    return (decades,)


@app.cell
def _(PROCESSING_DIR, decades):
    files_for_decades = {}

    for ix in decades.keys():
        start_dec, end_dec = decades[ix]
        _files_for_decade = list(PROCESSING_DIR.glob(f'*{start_dec}-{end_dec}*.nc'))
        files_for_decades[ix] = _files_for_decade

    # Uncomment this to see values for debugging
    # the1950s = files_for_decades['1950']
    # the1950s
    return (files_for_decades,)


@app.cell
def _(DECADAL_DIR, files_for_decades, xr):
    # This doesn't break because it's a lazy-load
    # the1950s_all = [xr.open_dataset(f) for f in the1950s[:40]]

    decade_of_interest = '1990'
    files_for_decade = files_for_decades[decade_of_interest]
    groupings = [files_for_decade[i:i + 40] for i in range(0, len(files_for_decade), 40)]
    print(f"{len(groupings)} file groupings to be used for decade {decade_of_interest}")
    for i, grouping in enumerate(groupings):
        loaded = [xr.open_dataset(f) for f in grouping]
        print(f"Loaded group {i}")
        combined = xr.concat(loaded, dim='report', data_vars='all')
        combined['reporting_stats'] = combined['reporting_stats'].fillna(-999.0)
        # combined = combined.chunk(time=xr.groupers.TimeResampler("MS"))
        print(f"Combined group {i}")
        filename = f'all_{decade_of_interest}s_group{str(i)}.nc'
        combined.to_netcdf(DECADAL_DIR / filename)
        print(f"Wrote group {i}")
    return (combined,)


@app.cell
def _():
    print('donezo all')
    return


@app.cell
def _(combined):
    combined.sel({'time': '1990-01-01'}).temperatures.plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
