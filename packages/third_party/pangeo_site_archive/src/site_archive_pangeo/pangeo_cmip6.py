"""
Pangeo CMIP6 accessor.

Reads the Pangeo analysis-ready, cloud-optimised (ARCO) CMIP6 data, which is published as one Zarr store
per variable and described by a single catalog CSV.
"""

import os
import warnings
from typing import Any, ClassVar

import pandas
import pyearthtools.data
import xarray
from pyearthtools.data import Petdt
from pyearthtools.data.archive import register_archive
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.transforms import Transform, TransformCollection

DEFAULT_CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"

_CATALOG_COLUMNS = [
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
    "zstore",
    "version",
]


@register_archive(
    "PangeoCMIP6",
    sample_kwargs={"variables": ["tas"], "source_id": "CanESM5", "experiment_id": "historical"},
)
class PangeoCMIP6(ArchiveIndex):
    """
    Data accessor for the Pangeo CMIP6 analysis-ready, cloud-optimised (ARCO) data.

    A catalog CSV lists the Zarr store of every variable of every model run. This accessor selects one
    model run (``source_id``, ``experiment_id``, ``member_id``, ``table_id``) and opens the store of each
    requested variable lazily, so only the parts of the data that are used are downloaded.

    A store holds all times, so ``filesystem`` ignores the requested time and the time is selected after
    loading.

    Examples:
        >>> index = PangeoCMIP6(["tas"], source_id="CanESM5", experiment_id="historical")
        >>> index.retrieve("2000-01")
    """

    _OPEN_ARGS: ClassVar[dict[str, Any]] = {"consolidated": True, "storage_options": {"token": "anon"}}

    def __init__(
        self,
        variables: list[str] | str,
        *,
        source_id: str,
        experiment_id: str,
        member_id: str = "r1i1p1f1",
        table_id: str = "Amon",
        grid_label: str | None = None,
        catalog: str | os.PathLike = DEFAULT_CATALOG,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Set up the Pangeo CMIP6 accessor.

        The catalog is not read here, only when data is first requested.

        Args:
            variables: Variables to retrieve, e.g. ``["tas", "pr"]``.
            source_id: Model name, e.g. ``"CanESM5"``.
            experiment_id: Experiment, e.g. ``"historical"`` or ``"ssp585"``.
            member_id: Ensemble member. Defaults to ``"r1i1p1f1"``.
            table_id: Frequency and realm table, e.g. ``"Amon"`` for monthly atmosphere. Defaults to ``"Amon"``.
            grid_label: Grid, e.g. ``"gn"`` (native) or ``"gr"`` (regridded). Needed when a variable exists
                on more than one grid. Defaults to None.
            catalog: Location of the Pangeo catalog CSV, a URL or a local path.
            transforms: Extra transforms to apply after loading.

        Raises:
            ValueError: If no variables are given.
        """
        self._variables = [variables] if isinstance(variables, str) else list(variables)
        if not self._variables:
            raise ValueError("At least one variable is required.")

        self._query = {
            "source_id": source_id,
            "experiment_id": experiment_id,
            "member_id": member_id,
            "table_id": table_id,
        }
        self._grid_label = grid_label
        self._catalog_src = catalog
        self._catalog = None  # read lazily, see _load_catalog

        extra_transforms = TransformCollection() if transforms is None else TransformCollection(transforms)
        super().__init__(
            transforms=TransformCollection([pyearthtools.data.transforms.variables.Trim(self._variables)])
            + extra_transforms,
        )
        self.record_initialisation()

    def _load_catalog(self) -> pandas.DataFrame:
        """Read the catalog once, keeping only the rows of the requested variables and model run."""
        if self._catalog is None:
            catalog = self.catalog(self._catalog_src)
            catalog = catalog[catalog["variable_id"].isin(self._variables)]
            for column, value in self._query.items():
                catalog = catalog[catalog[column] == value]
            self._catalog = catalog.reset_index(drop=True)
        return self._catalog

    def filesystem(self, querytime: str | Petdt) -> dict[str, str]:
        """
        Find the Zarr store of each variable.

        Args:
            querytime: Ignored, as a store holds all times.

        Raises:
            ValueError: If a variable has no store, exists on several grids and no ``grid_label`` was given,
                or has several stores at its latest version.

        Returns:
            Store URL of each variable, using the latest version in the catalog.
        """
        catalog = self._load_catalog()
        stores = {}
        for variable in self._variables:
            rows = catalog[catalog["variable_id"] == variable]
            if self._grid_label is not None:
                rows = rows[rows["grid_label"] == self._grid_label]
            if rows.empty:
                raise ValueError(f"No Pangeo CMIP6 store for variable {variable!r} matching {self._query}")
            grids = sorted(rows["grid_label"].unique())
            if len(grids) > 1:
                raise ValueError(
                    f"Variable {variable!r} exists on several grids {grids}; pass grid_label to choose one."
                )
            latest = rows[rows["version"] == rows["version"].max()]
            urls = sorted(latest["zstore"].unique())
            if len(urls) > 1:
                raise ValueError(
                    f"Variable {variable!r} has several stores at the latest version {urls}; the run is ambiguous."
                )
            stores[variable] = str(urls[0])
        return stores

    def load(self, files: dict[str, str], **kwargs) -> xarray.Dataset:
        """
        Open the store of each variable lazily, keep only that variable, and merge them.

        Args:
            files: Store URL of each variable, as found by ``filesystem``.
            **kwargs: Passed to ``xarray.open_zarr``.

        Returns:
            Dataset holding all times of all the variables. Scalar coordinates that differ between the variables
            are dropped, e.g. ``height``, which is 2 m for ``tas`` but 10 m for ``uas``.
        """
        open_args = {**self._OPEN_ARGS, **kwargs}
        datasets = [xarray.open_zarr(url, **open_args)[[variable]] for variable, url in files.items()]
        return xarray.merge(datasets, join="exact", compat="minimal")

    def retrieve(self, querytime: str | Petdt, *args, **kwargs) -> xarray.Dataset:
        """
        Retrieve the data at ``querytime``.

        Raises:
            DataNotFoundError: If the store holds no data at ``querytime``. Without this, the parent class
                only warns and returns every time of the store.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = super().retrieve(querytime, *args, **kwargs)
        for warning in caught:
            if "Could not find time in dataset" in str(warning.message):
                raise pyearthtools.data.DataNotFoundError(f"No data found at {querytime}.")
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
        return data

    @staticmethod
    def catalog(catalog: str | os.PathLike = DEFAULT_CATALOG) -> pandas.DataFrame:
        """Return the Pangeo CMIP6 catalogue, so the available data can be discovered.

        Args:
            catalog: CSV catalogue to read. Defaults to the public Pangeo CMIP6 catalogue.

        Returns:
            DataFrame with one row per variable of one model run, with the columns `source_id`,
            `experiment_id`, `member_id`, `table_id`, `variable_id`, `grid_label`, `zstore` and `version`.
        """
        return pandas.read_csv(catalog, usecols=_CATALOG_COLUMNS)

    def exists(self, *args, **kwargs) -> bool:
        """Whether a store is found for every variable. Stores are URLs, so they are not checked on disk."""
        try:
            self.search(*args)
        except ValueError:
            return False
        return True

    @property
    def _desc_(self):
        return {
            "singleline": "Pangeo CMIP6 analysis-ready, cloud-optimised data accessor",
            "Documentation": "https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html",
        }
