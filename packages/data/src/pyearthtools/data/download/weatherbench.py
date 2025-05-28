import xarray as xr

import pyearthtools.data
from pyearthtools.data.time import Petdt

from pyearthtools.data.indexes import AdvancedTimeDataIndex, decorators
from pyearthtools.data.transforms.transform import Transform, TransformCollection


#: valid WeatherBench level values
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

#: mapping from long variable names to short variable names
LONG_NAMES = {
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

#: mapping from short variable names to long variable names
SHORT_NAMES = {val: key for key, val in LONG_NAMES.items() if val is not None}


def open_weatherbench2(variables, level=None, chunks="auto", **kwargs):
    """Open a WeatherBench2 dataset from Google Cloud Platform"""

    # skip parsing unused variables, this can make loading much faster
    drop_variables = [var for var in LONG_NAMES if var not in set(variables)]

    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        chunks=chunks,
        storage_options=dict(token="anon"),
        drop_variables=drop_variables,
        **kwargs,
    )

    if level is not None:
        ds = pyearthtools.data.transform.coordinates.Select(level=level, ignore_missing=True)(ds)

    return ds


_VALID_LEVELS = [None] + LEVELS
_VALID_VARIABLES = [None] + list(LONG_NAMES) + list(SHORT_NAMES)


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
    @decorators.check_arguments(variables=_VALID_VARIABLES, level=_VALID_LEVELS)
    @decorators.variable_modifications("variables")
    def __init__(
        self,
        variables: str | list[str] | None = None,
        level: int | list[int] | None = None,
        transforms: Transform | TransformCollection | None = None,
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

        # load all variables by default
        if variables is None:
            variables = list(LONG_NAMES)

        if not isinstance(variables, list):
            variables = [variables]

        # convert variable name if found in short name mapping
        variables = [SHORT_NAMES.get(var, var) for var in variables]

        self.variables = variables
        self.level = level

        self._kwargs = kwargs
        self._ds = open_weatherbench2(variables, level, **kwargs)

    @property
    def dataset(self) -> xr.Dataset:
        """Get full dataset for this obj"""
        return self._ds

    def get(self, time: str):
        """Get timestep from dataset"""
        return self._ds.sel(time=Petdt(time).datetime64())

    @classmethod
    def sample(cls):
        """Example subset of the dataset"""
        return WeatherBench2("2m_temperature")
