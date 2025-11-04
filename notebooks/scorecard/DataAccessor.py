import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():

    import pyearthtools.data
    import pyearthtools.pipeline
    from pyearthtools.data import Petdt

    from pathlib import Path
    DECADAL_DIR = Path.home() / 'hadisd' / 'by_decade'

    from mpl_toolkits.basemap import Basemap
    return Basemap, DECADAL_DIR, Path, Petdt, pyearthtools


@app.cell
def _(Path, Petdt):
    from pyearthtools.data.archive import register_archive
    from pyearthtools.data.exceptions import DataNotFoundError
    from pyearthtools.data.indexes import ArchiveIndex, decorators
    from pyearthtools.data.transforms import Transform, TransformCollection
    import xarray as xr
    import numpy as np

    @register_archive("ISD", sample_kwargs=dict(variable="2t"))
    class ISD(ArchiveIndex):
        @property
        def _desc_(self):
            return {
                "singleline": "Hadley Integrated Surface Database",
                "range": "1930 - 2025",
                "Documentation": "https://www.metoffice.gov.uk/hadobs/hadisd/",
            }

        def __init__(
            self,
            disk_location,
            *,        
            transforms: Transform | TransformCollection | None = None,
        ):

            self.disk_location = Path(disk_location)  # Location of the large groupings files
            super().__init__(transforms=transforms or TransformCollection())

        def filesystem(self, querytime: str | Petdt):
            '''
            This is quick, no need to cache it
            '''
            files = list(self.disk_location.glob('*1990*.nc'))
            return files

        def load(self, from_files_list, **kwargs):

            ds = xr.open_mfdataset(from_files_list, combine='nested', concat_dim='report')

            # Arguably, this should be a transform, or handled in the pipeline, but it works for now
            ds['temperatures'] = ds.temperatures.where(ds.temperatures > -1000)
            return ds
    return ISD, np


@app.cell
def _(DECADAL_DIR, ISD):
    stations = ISD(DECADAL_DIR)
    return (stations,)


@app.cell
def _(stations):
    ds = stations['1990-06-20T01']
    ds
    return (ds,)


@app.cell
def _(Petdt):
    Petdt('1990-06-20').datetime.year
    return


@app.cell
def _(ds):
    import plotly.express as px

    px.scatter(ds.temperatures.values[0])
    return


@app.cell
def _(DECADAL_DIR, ISD, Path, pyearthtools):
    workdir = Path("~/dev/data/wb2era5/")
    era5_source = pyearthtools.data.download.weatherbench.WB2ERA5(
            variables=["2m_temperature", "u", "v", "geopotential"],
            level=[850],
            download_dir=workdir / "download",
            license_ok=True,
        ),

    station_source = ISD(DECADAL_DIR)

    data_pipeline = pyearthtools.pipeline.Pipeline(
        (era5_source, station_source)
    )
    return (data_pipeline,)


@app.cell
def _(data_pipeline):
    data_pipeline['19900620T00']
    return


@app.cell
def _(data_pipeline):
    grid, points = data_pipeline['19900620T00']
    return grid, points


@app.cell
def _(grid, np):
    # Transform gridded data for plotting
    lats = grid['latitude'].values
    lons = grid['longitude'].values
    data = grid['2m_temperature'].values[0]  # Replace with your variable name
    lon, lat = np.meshgrid(lons, lats)
    return data, lat, lon


@app.cell
def _(Basemap, data, lat, lon, points):
    map = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
                llcrnrlon=0,urcrnrlon=360,lat_ts=20,resolution='l')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)

    x, y = map(lon, lat)


    # # Add station data over the top
    x2, y2 = map(points.lon, points.lat)

    map.contourf(x, y, data.T, cmap='viridis')
    map.scatter(x2, y2, c=points.temperatures, cmap='viridis')
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
