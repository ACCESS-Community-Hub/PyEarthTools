import textwrap
import hashlib
from pathlib import Path
from typing import Literal

import xarray as xr
from numcodecs.blosc import Blosc

from pyearthtools.data.time import Petdt
from pyearthtools.data.indexes import AdvancedTimeDataIndex, decorators
from pyearthtools.data.indexes.utilities import spellcheck
from pyearthtools.data.transforms.transform import Transform, TransformCollection
from pyearthtools.data.transforms.coordinates import Select


def _extract_dataset_infos(url: str) -> tuple[dict[str, str | None], list[int]]:
    """helper function to extract variable mapping and levels from a given dataset"""
    dset = xr.open_zarr(url)
    long_names = {vname: dset[vname].attrs.get("short_name") for vname in dset}
    levels = sorted(set(dset.level.values.tolist()))
    return long_names, levels


def create_dataset_mapping(module_path: str):
    """generate dataset infos mapping for a set of urls and save it as a module"""
    urls = [
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_512x256_equiangular_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_512x256_equiangular_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr",
        "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr",
    ]
    dataset_infos = {url: _extract_dataset_infos(url) for url in urls}
    module_txt = f"""\
    \"\"\"WeatherBench2 datasets validation information

    This module has been generated automatically using the function
    `pyearthtools.data.download.weatherbench.create_dataset_mapping`.
    Do not modify manually.
    \"\"\"

    #: infos for WeatherBench2 datasets, mapping urls to 2 elements tuples made of
    #: - a mapping from long variable names to short variable names
    #: - valid level values
    DATASETS_INFOS = {dataset_infos!r}
    """
    Path(module_path).write_text(textwrap.dedent(module_txt))


def save_local_dataset(path: Path, dset: xr.Dataset):
    """save a dataset as a set of local .zarr folders, one per variable and level

    Note: Variables already saved in `path` are skipped.
    """
    # TODO add logging

    path.mkdir(parents=True, exist_ok=True)

    for varname in dset.data_vars:
        dset_var = dset[varname]
        compressor = {"compressor": Blosc(cname="zstd", clevel=6)}
        zarr_kwargs = {"encoding": {varname: compressor}, "consolidated": False}

        if "level" in dset_var.dims:
            for level in dset_var.level.values:
                zarrpath = path / f"{varname}_level-{level}.zarr"
                if not zarrpath.is_dir():
                    dset_var.sel(level=level).to_zarr(zarrpath, **zarr_kwargs)

        else:
            zarrpath = path / f"{varname}.zarr"
            if not zarrpath.is_dir():
                dset_var.to_zarr(zarrpath, **zarr_kwargs)


class MissingVariableFile(FileNotFoundError):
    pass


def open_local_dataset(path: Path, variables: list[str], level: list[int]) -> xr.Dataset:
    # TODO reopen with right chunking

    dsets = []

    for varname in variables:
        filepath = path / f"{varname}.zarr"

        if filepath.is_dir():
            dset = xr.open_zarr(filepath, consolidated=False)
        else:
            filelist = [path / f"{varname}_level-{lvl}.zarr" for lvl in level]
            if any(not fpath.is_dir() for fpath in filelist):
                raise MissingVariableFile("Missing .zarr folder for some variables")
            dset = xr.open_mfdataset(filelist, concat_dim="level", combine="nested", consolidated=False)

        dsets.append(dset)

    dset_full = xr.merge(dsets)
    return dset_full


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
        download_dir: str | Path | None = None,
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
            chunks (int | dict | Literal["auto"], optional):
                Chunking used to load data into Dask arrays. Defaults to "auto".
        """
        super().__init__(transforms or TransformCollection(), data_interval="1 hour")
        self.record_initialisation()

        # retrieve variables name mapping and levels for the dataset
        from pyearthtools.data.download._weatherbench import DATASETS_INFOS

        long_names, valid_levels = DATASETS_INFOS[url]

        # create short variables name mappings
        short_names = {val: key for key, val in long_names.items() if val is not None}

        # check variables and level values
        if variables is not None:
            valid_variables = list(long_names) + list(short_names)
            spellcheck.check_prompt(variables, valid_variables, name="variables")

        if level is not None:
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

        def open_online_dataset():
            # skip parsing unused variables, this can make loading much faster
            drop_variables = [var for var in long_names if var not in set(variables)]
            ds = xr.open_zarr(url, chunks=chunks, drop_variables=drop_variables, **kwargs)
            if level is not None:
                ds = Select(level=level, ignore_missing=True)(ds)
            return ds

        if download_dir is None:
            ds = open_online_dataset()

        else:
            # use a hash of the url to identify the dataset subfolder
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            download_path = Path(download_dir) / url_hash

            # try to open dataset from download dir if defined
            # download missing variables and levels if this fails
            try:
                ds = open_local_dataset(download_path, variables, level)
            except MissingVariableFile:
                ds_remote = open_online_dataset()
                save_local_dataset(download_path, ds_remote)
                (download_path / "dataset_url").write_text(url)
                ds = open_local_dataset(download_path, variables, level)

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
