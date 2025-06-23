from typing import Literal

import xarray as xr

from pyearthtools.data.time import Petdt

from pyearthtools.data.indexes import AdvancedTimeDataIndex, decorators
from pyearthtools.data.indexes.utilities import spellcheck
from pyearthtools.data.transforms.transform import Transform, TransformCollection
from pyearthtools.data.transforms.coordinates import Select


DATASETS_LONG_NAMES = {
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr": {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "10m_wind_speed": None,
        "2m_dewpoint_temperature": "d2m",
        "2m_temperature": "t2m",
        "above_ground": None,
        "ageostrophic_wind_speed": None,
        "angle_of_sub_gridscale_orography": "anor",
        "anisotropy_of_sub_gridscale_orography": "isor",
        "boundary_layer_height": "blh",
        "divergence": None,
        "eddy_kinetic_energy": None,
        "geopotential": "z",
        "geopotential_at_surface": "z",
        "geostrophic_wind_speed": None,
        "high_vegetation_cover": "cvh",
        "integrated_vapor_transport": None,
        "lake_cover": "cl",
        "land_sea_mask": "lsm",
        "lapse_rate": None,
        "leaf_area_index_high_vegetation": "lai_hv",
        "leaf_area_index_low_vegetation": "lai_lv",
        "low_vegetation_cover": "cvl",
        "mean_sea_level_pressure": "msl",
        "mean_surface_latent_heat_flux": "mslhf",
        "mean_surface_net_long_wave_radiation_flux": "msnlwrf",
        "mean_surface_net_short_wave_radiation_flux": "msnswrf",
        "mean_surface_sensible_heat_flux": "msshf",
        "mean_top_downward_short_wave_radiation_flux": "mtdwswrf",
        "mean_top_net_long_wave_radiation_flux": "mtnlwrf",
        "mean_top_net_short_wave_radiation_flux": "mtnswrf",
        "mean_vertically_integrated_moisture_divergence": "mvimd",
        "potential_vorticity": "pv",
        "relative_humidity": None,
        "sea_ice_cover": "siconc",
        "sea_surface_temperature": "sst",
        "slope_of_sub_gridscale_orography": "slor",
        "snow_depth": "sd",
        "soil_type": "slt",
        "specific_humidity": "q",
        "standard_deviation_of_filtered_subgrid_orography": "sdfor",
        "standard_deviation_of_orography": "sdor",
        "surface_pressure": "sp",
        "temperature": "t",
        "total_cloud_cover": "tcc",
        "total_column_vapor": None,
        "total_column_water": "tcw",
        "total_column_water_vapour": "tcwv",
        "total_precipitation_12hr": "tp",
        "total_precipitation_24hr": "tp",
        "total_precipitation_6hr": "tp",
        "type_of_high_vegetation": "tvh",
        "type_of_low_vegetation": "tvl",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "vertical_velocity": "w",
        "volumetric_soil_water_layer_1": "swvl1",
        "volumetric_soil_water_layer_2": "swvl2",
        "volumetric_soil_water_layer_3": "swvl3",
        "volumetric_soil_water_layer_4": "swvl4",
        "vorticity": None,
        "wind_speed": None,
    }
}

DATASETS_LEVELS = {
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr": [
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    ]
}


class WeatherBench2(AdvancedTimeDataIndex):
    """WeatherBench2 cloud-optimized ground truth and baseline datasets

    https://github.com/google-research/weatherbench2

    Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia,
    Tyler Russel, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal,
    Matthew Chantry, Zied Ben Bouallegue, Peter Dueben, Carla Bromberg, Jared Sisk,
    Luke Barrington, Aaron Bell and Fei Sha (2024):
    WeatherBench 2: A benchmark for the next generation of data-driven global
    weather models
    Journal of Advances in Modeling Earth Systems, 16, e2023MS004019
    https://doi.org/10.1029/2023MS004019
    """

    _desc_ = {
        "singleline": "WeatherBench2 cloud-optimized ground truth and baseline datasets",
        "link": "https://github.com/google-research/weatherbench2",
    }

    @decorators.alias_arguments(variables=["variable"], level=["levels", "level_value"])
    @decorators.variable_modifications("variables")
    def __init__(
        self,
        url: str,
        *,
        variables: str | list[str] | None = None,
        level: int | list[int] | None = None,
        transforms: Transform | TransformCollection | None = None,
        chunks: int | dict | Literal["auto"] | None = "auto",
        **kwargs,
    ):
        """WeatherBench2 cloud-optimized datasets integrated within `pyearthtools`

        Allows for access to a dataset for WeatherBench2 collection.

        Args:
            variables (str | list[str] | None, optional):
                Variables to retrieve, can be either short_name or long_name.
                Default to None, to retrieve all variables.
            level (int | list[int] | None, optional):
                Pressure levels to select. Defaults to None, to select all levels.
            transforms (Transform | TransformCollection | None, optional):
                Transforms to apply to dataset. Defaults to None.
        """
        super().__init__(transforms or TransformCollection(), data_interval="1 hour")
        self.record_initialisation()

        # retrieve long and short variables name mappings
        long_names = DATASETS_LONG_NAMES[url]
        short_names = {val: key for key, val in long_names.items() if val is not None}

        # check variables and level values
        if variables is not None:
            valid_variables = list(long_names) + list(short_names)
            spellcheck.check_prompt(variables, valid_variables, name="variables")

        if level is not None:
            valid_levels = DATASETS_LEVELS[url]
            spellcheck.check_prompt(level, valid_levels, name="level")

        # load all variables by default
        if variables is None:
            variables = list(long_names)

        if not isinstance(variables, list):
            variables = [variables]

        # convert variable name if found in short name mapping
        variables = [short_names.get(var, var) for var in variables]

        self.variables = variables
        self.level = level

        # skip parsing unused variables, this can make loading much faster
        drop_variables = [var for var in long_names if var not in set(variables)]
        ds = xr.open_zarr(url, chunks=chunks, drop_variables=drop_variables, **kwargs)
        if level is not None:
            ds = Select(level=level, ignore_missing=True)(ds)

        self._ds = ds
        self._kwargs = kwargs

    @property
    def dataset(self) -> xr.Dataset:
        """Get full dataset for this obj"""
        return self._ds

    def get(self, time: str):
        """Get timestep from dataset"""
        return self._ds.sel(time=Petdt(time).datetime64())


class WB2ERA5(WeatherBench2):
    """WeatherBench2 cloud-optimized ground truth ERA5 dataset

    ERA5 datasets downloaded from the Copernicus Climate Data Store with a time
    range from 1959 to 2023 (incl.). The data have been downsampled to 6h and
    13 levels, except for the "raw" dataset. The raw dataset is hourly with a
    0.25 degree spatial resolution and 37 levels.

    https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5

    Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia,
    Tyler Russel, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal,
    Matthew Chantry, Zied Ben Bouallegue, Peter Dueben, Carla Bromberg, Jared Sisk,
    Luke Barrington, Aaron Bell and Fei Sha (2024):
    WeatherBench 2: A benchmark for the next generation of data-driven global
    weather models
    Journal of Advances in Modeling Earth Systems, 16, e2023MS004019
    https://doi.org/10.1029/2023MS004019
    """

    _desc_ = {
        "singleline": "WeatherBench2 cloud-optimized ground truth ERA5 dataset",
        "link": "https://github.com/google-research/weatherbench2",
    }

    DATASETS = {
        "raw": "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        "1440x721": "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        "240x121": "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "64x32": "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    }

    @decorators.check_arguments(resolution=["1440x721", "240x121", "64x32"])
    def __init__(self, resolution: str = "64x32", **kwargs):
        url = f"gs://weatherbench2/datasets/era5/{self.DATASETS[resolution]}"
        super().__init__(url, **kwargs)
        self.resolution = resolution

    @classmethod
    def sample(cls):
        """Example subset of the dataset"""
        return WB2ERA5("64x32", variables="2m_temperature")


class WB2ERA5Clim(WeatherBench2):
    """WeatherBench2 cloud-optimized ground truth ERA5 climatology dataset

    For WeatherBench 2, the climatology was computed using a running window for
    smoothing (see paper and script) for each day of year and sixth hour of day.
    Climatologies have been computed for 1990-2017 and 1990-2019.

    https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5-climatology

    Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia,
    Tyler Russel, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal,
    Matthew Chantry, Zied Ben Bouallegue, Peter Dueben, Carla Bromberg, Jared Sisk,
    Luke Barrington, Aaron Bell and Fei Sha (2024):
    WeatherBench 2: A benchmark for the next generation of data-driven global
    weather models
    Journal of Advances in Modeling Earth Systems, 16, e2023MS004019
    https://doi.org/10.1029/2023MS004019
    """

    DATASETS = {
        ("1990-2017", "1440x721"): "1990-2017_6h_1440x721.zarr",
        ("1990-2017", "512x256"): "1990-2017_6h_512x256_equiangular_conservative.zarr",
        ("1990-2017", "240x121"): "1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr",
        ("1990-2017", "64x32"): "1990-2017_6h_64x32_equiangular_conservative.zarr",
        ("1990-2019", "1440x721"): "1990-2019_6h_1440x721.zarr",
        ("1990-2019", "512x256"): "1990-2019_6h_512x256_equiangular_conservative.zarr",
        ("1990-2019", "240x121"): "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr",
        ("1990-2019", "64x32"): "1990-2019_6h_64x32_equiangular_conservative.zarr",
    }

    @decorators.check_arguments(resolution=["1440x721", "240x121", "64x32"], period=["1990-2017", "1990-2019"])
    def __init__(self, resolution: str = "64x32", period: str = "1990-2017", **kwargs):
        fname = self.DATASETS[(self.period, self.resolution)]
        url = f"gs://weatherbench2/datasets/era5-hourly-climatology/{fname}"
        super().__init__(url, **kwargs)
        self.period = period
        self.resolution = resolution

    @classmethod
    def sample(cls):
        """Example subset of the dataset"""
        return WB2ERA5Clim("64x32", "1990-2017", "2m_temperature")
