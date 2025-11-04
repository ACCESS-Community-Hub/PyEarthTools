import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import xarray as xr
    from pathlib import Path

    DECADAL_DIR = Path.home() / 'hadisd' / 'by_decade'    # This will hold the final form of our data
    return DECADAL_DIR, xr


@app.cell
def _(DECADAL_DIR):
    list(DECADAL_DIR.glob('*1990s*.nc'))
    return


@app.cell
def _(DECADAL_DIR, xr):
    ds = xr.open_dataset(list(DECADAL_DIR.glob('*1990s*.nc'))[3])
    ds


    return (ds,)


@app.cell
def _():

    # ds = xr.open_mfdataset(list(DECADAL_DIR.glob('*1990s*.nc')), combine='nested', concat_dim='report')
    # ds
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return


@app.cell
def _(ds):
    sample = ds.sel({'time': '1990-06-01T00'})
    sample = sample.assign_coords({'report': sample.report})
    return (sample,)


@app.cell
def _(sample):
    sample.sel({'report': 0})
    return


@app.cell
def _(sample):
    sample.report
    return


@app.cell
def _(sample):

    import folium

    m = folium.Map(location=(45.5236, -122.6750))
    import numpy as np

    for report in sample.report:
        lat = sample.sel({'report': report}).lat.values
        lon = sample.sel({'report': report}).lon.values

        if not np.isnan(lat):

            try:
    
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(icon="cloud"),
                ).add_to(m)

            except:
                print(lat)

                raise

    return (m,)


@app.cell
def _(m):
    m
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
