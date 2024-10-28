# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Intake - ESM Index
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import xarray as xr
import logging
import functools

from edit.data.indexes.indexes import DataIndex
from edit.data.indexes.cacheIndex import FileSystemCacheIndex
from edit.data.transforms import Transform, TransformCollection
from edit.data.patterns.argument import flattened_combinations

from edit.data.indexes.utilities.delete_files import delete_path

LOG = logging.getLogger("edit.data")


class IntakeIndex(DataIndex):
    """
    Index designed to operate on Intake ESM Catalogs

    Will not cache the data anywhere.

    ## Example:
    ```python
    import edit.data
    import intake_esm

    cat_url = intake_esm.tutorial.get_url("google_cmip6")

    intakeIndex = edit.data.IntakeIndex(cat_url)
    intakeIndex(experiment_id=["historical", "ssp585"],table_id="Oyr",variable_id="o2",grid_label="gn")
    ```
    """

    @property
    def _desc_(self):
        return {
            "singleline": "Intake ESM Catalog",
        }

    def __init__(
        self,
        catalog_file: str | Path,
        transforms: Transform | TransformCollection = TransformCollection(),
        *,
        add_default_transforms: bool = True,
        filter_dict: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Intake ESM Catalog Index

        Args:
            catalog_file (str | Path):
                Intake ESM Catalog location
            transforms (Transform | TransformCollection, optional):
                Transforms to add to data. Defaults to TransformCollection().
            add_default_transforms (bool, optional):
                Add default transforms. Defaults to True.
            filter_dict (dict, optional):
                Filter dictionary for `Intake` search. Defaults to {}.
        Raises:
            ImportError:
                If `intake` cannot be imported.
        """
        super().__init__(transforms, add_default_transforms=add_default_transforms)
        self._catalog_file = catalog_file
        self.record_initialisation()

        try:
            import intake

            self._intake_catalog = intake.open_esm_datastore(catalog_file)  # type: ignore
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            raise ImportError("Could not import `intake`, and access `intake-esm`, ensure they are installed.") from e

        self._search_kwargs: dict[str, Any] = {}
        self.update_filter(filter_dict or {}, **kwargs)

    @property
    def filter(self) -> dict[str, Any]:
        """
        Get filters applied to data retrieval

        Returns:
            (dict):
                Intake ESM search kwargs
        """
        return self._search_kwargs

    @property
    def intake(self):
        return self._intake_catalog

    def update_filter(self, filter_dict: dict[str, Any] | None = None, **kwargs) -> None:
        """
        Update filter for intake searching

        Args:
            filter_dict (dict[str, Any], optional):
                Filter update. Defaults to {}.
        """
        filter_dict = filter_dict or {}
        filter_dict.update(kwargs)
        self._search_kwargs.update(filter_dict)

    def pop_filter(self, pop: list[str] = [], *args: str) -> None:
        """
        Pop filter elements from intake searching

        Args:
            pop (list[str], optional):
                Items to pop from filter
            *args (str, optional):
                Args form of pop.
        """
        if not isinstance(pop, (list, tuple)):
            pop = [pop]

        tuple(pop.append(a) for a in args)

        for p in pop:
            self._search_kwargs.pop(p)

    def search_intake(self, filter_dict: dict[str, Any] = {}, **kwargs: Any) -> "intake_esm.source.ESMDataSource":  # type: ignore  # noqa: F821
        """
        Search Intake Catalog

        Uses `filter` set through `init` and `update_filter`

        Args:
            filter_dict (dict, optional):
                Updates to filters. Defaults to {}.

        Returns:
            (intake_esm.core.esm_datastore):
                Intake catalog after search
        """
        filter = dict(self._search_kwargs)
        filter.update(**filter_dict, **kwargs)
        return self._intake_catalog.search(**filter)  # type: ignore

    def search(self, filter: dict[str, Any] = {}, **kwargs: Any) -> "intake_esm.source.ESMDataSource":  # type: ignore  # noqa: F821
        """
        Override for Index search,

        As this is primarily an Intake Index, search Intake Catalog

        Uses `filter` set through `init` and `update_filter`, as will as those given here.

        Args:
            filter (dict[str, Any], optional):
                Intake search filter, updates filters given in `init`. Defaults to {}.
            kwargs (Any):
                Extra kwargs for `filter`.

        Returns:
            (intake_esm.core.esm_datastore):
                Intake catalog after search
        """
        return self.search_intake(filter, **kwargs)

    def _get_from_intake(
        self, filter: dict[str, Any] = {}, merge: bool = True, **kwargs: Any
    ) -> xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray]:
        """
        Get data from Intake ESM Catalog

        Args:
            filter (dict[str, Any], optional):
                Intake search filter, updates filters given in `init`. Defaults to {}.
            merge (bool, optional):
                Whether to attempt to merge data. If fails, return data prior to merge. Defaults to True.
            kwargs (Any, optional):
                Extra filters to apply.

        Returns:
            (xr.Dataset | dict[str, xr.Dataset]):
                Data from catalog in dict, or if can merge, `xr.Dataset`.
                If `merge == False`, will be dict.

        Raises:
            KeyError:
                If searching yields no results.

        """
        ds_dict: dict[str, xr.Dataset] = self.search_intake(filter, **kwargs).to_dataset_dict()

        if len(ds_dict) == 0:
            raise KeyError(f"Searching with filter, {kwargs} & {self._search_kwargs} yielded no results.")

        if not merge:
            return ds_dict  # type: ignore

        try:
            return xr.combine_by_coords(ds_dict.values())
        except Exception as e:
            LOG.info(f"Failed to combine_by_coords data with keys: {ds_dict.keys()}. {e}")
        return ds_dict  # type: ignore

    def get(self, **kwargs):
        """
        Get data directly from `intake`

        See `._get_from_intake` for docs.
        """
        return self._get_from_intake(**kwargs)

    def __getitem__(self, idx: Any) -> Any:
        """
        getitem override for Intake ESM

        First tries to retrieve it from the catalog, and fails over to default `Index` behaviour.

        Args:
            idx (Any):
                Idx to retrieve

        Returns:
            (Any):
                Result of `__getitem__` either on Intake catalog or self

        """
        try:
            return self._intake_catalog[idx]  # type: ignore
        except Exception as e:
            LOG.debug(f"Cannot find {idx} in intake catalog, default to super() method. {e}")
        return super().__getitem__(idx)


class IntakeIndexCache(FileSystemCacheIndex, IntakeIndex):
    """
    Intake ESM Index which caches to a local location.

    Uses `ArgumentExpansion` in the same order as the catalog itself.

    Effectively builds a local copy of the intake catalog.

    !!! note "Multiple Keys":
        As the data is saved according to the given filters, list or tuples in the filters will be split during the `filesystem` search,
        and handled one after the other.
        This will cause the underlying pattern to not be exactly usable as a cache, elements for it will have to be atomic.
    """

    def __init__(
        self,
        catalog_file: str | Path,
        cache: str | Path | None = None,
        pattern_kwargs: dict[str, Any] | None = None,
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
        filter_dict: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Caching Intake ESM Index

        Args:
            catalog_file (str | Path):
                Intake ESM Catalog to load.
            cache (str | Path | None, optional):
                Cache Location. If set to None, does not cache. Defaults to None.
            filter_dict (dict, optional):
                Default filters for searching the Intake ESM Catalog. Defaults to {}.
            **kwargs (Any, optional):
                Additional filters.

        See `edit.data.indexes.BaseCacheIndex` for remaining arguments docs.
        """
        pattern_kwargs = pattern_kwargs or {}

        pattern_kwargs["expand_tuples"] = True
        super().__init__(
            cache,
            pattern="ArgumentExpansion",
            pattern_kwargs=pattern_kwargs or {},
            transforms=transforms,
            # Intake kwargs
            catalog_file=catalog_file,
            filter_dict=filter_dict or {},
            **kwargs,
        )
        self.record_initialisation()

    def _generate(self, **kwargs: Any) -> xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray]:
        """
        Get data from Intake ESM Catalog
        """
        return super()._get_from_intake(**kwargs)

    def _convert_to_args(self, key: str, df: "intake_esm.source.ESMDataSource", **kwargs: Any) -> list[str]:  # type: ignore # noqa: F821
        """
        Convert Intake ESM key to arguments for `ArgumentExpansion`

        Args:
            key (str):
                Base level key seperated by '.'
            df (intake_esm.source.ESMDataSource):
                Intake dataframe for key info retrieval
            kwargs (Any):
                Extra filter kwargs to add to args.

        Returns:
            (list[str]):
                List of arguments.
        """

        filters = dict(self.filter)
        filters.update(dict(kwargs))

        sorted_keys = list(set(filters.keys()).difference(set(df.keys_info().columns)))
        sorted_keys.sort()

        key_elements: list[str] = key.split(".")
        if "file_type" in df.keys_info().columns:
            key_elements.pop(list(df.keys_info().columns).index("file_type"))

        for k in sorted_keys:
            str_form = str(filters[k])
            if isinstance(k, (list, tuple)):
                str_form = "_".join(filters[k])
            key_elements.append(str_form)

        key_elements.insert(0, key_elements.pop())
        return key_elements

    def get(self, **kwargs: Any) -> xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray]:
        """
        Retrieve Data given filter kwargs

        If cache is given, automatically check to see if the file is generated,
        else, generate it and return the data

        If cache is not given, just generate and return the data

        Args:
            **kwargs (Any):
                Kwargs to generate with

        Returns:
            (xr.Dataset | dict[str, xr.Dataset]):
                Loaded data
        """

        if self.cache is None and self.pattern_type is None:
            return self._generate(**kwargs)

        return self.generate(**kwargs)

    def _split_queries(function: Callable) -> Callable:  # type: ignore
        """
        Split queries to the `filesystem` function into single str values from list | tuple.
        """

        def wrapper(self: IntakeIndexCache, **kwargs: Any):
            full_filter_kwargs = dict(self._search_kwargs)
            full_filter_kwargs.update(kwargs)
            if not any([isinstance(x, (list, tuple)) for x in kwargs.values()]):
                function(self, **full_filter_kwargs)

            keys = list(kwargs.keys())
            values = list(kwargs.values())

            results: list[str | Path] = []

            for perm in flattened_combinations(values):
                results.extend(function(self, **{keys[i]: perm[i] for i in range(len(perm))}))
            return results

        return wrapper

    @_split_queries
    def generate(self, **kwargs: Any) -> xr.Dataset | dict[str, xr.Dataset]:
        """
        Get data from the Intake catalog, and save it out based on the filter args.

        If any data is cached, will not override.

        Returns:
            (list[str | Path]):
                Location of cached data
        """
        pattern = self.pattern

        data_df = self.search_intake(**kwargs)
        if len(data_df) == 0:
            raise KeyError(f"Searching with filter, {kwargs} & {self._search_kwargs} yielded no results.")

        # Check to see if data has been saved
        data_paths: list = []

        for key, val in data_df.items():
            key_elements = self._convert_to_args(key, data_df, **kwargs)
            if not self._check_if_exists(*key_elements):
                pattern.save(val.to_dask(), *key_elements)
            data_paths.append(pattern.search(*key_elements))

        return self.load(data_paths, soft_fail=True)

    @_split_queries
    def filesystem(self, **kwargs: Any) -> list[str | Path]:
        """
        Search for generated data if cache is given.

        If data does not exist yet, generate it, save it, and return the path to it

        Args:
            kwargs (Any):
                kwargs to filter by

        Returns:
            (list[str | Path]):
                Filepath/s to cached data

        Raises:
            RuntimeError:
                If `cache` is not set, cannot cache data.
        """
        self.save_record()

        if self.cache is None and self.pattern_type is None:
            raise RuntimeError("`CachingIndex` cannot save data without a `cache` location.")

        pattern = self.pattern

        data_df = self.search_intake(**kwargs)  # type: ignore
        if len(data_df) == 0:
            raise KeyError(f"Searching with filter, {kwargs} & {self._search_kwargs} yielded no results.")

        # Check to see if data has been saved
        data_paths: list = []
        all_cached = True

        all_elements = list(self._convert_to_args(key, data_df, **kwargs) for key, _ in data_df.items())
        print(all_elements)
        for elem in all_elements:
            if not self._check_if_exists(*elem):
                all_cached = False
                break
            data_paths.append(pattern.search(*elem))

        if all_cached:
            return data_paths

        self.generate(**kwargs)
        return list(pattern.search(elem) for elem in all_elements)  # type: ignore

    @functools.wraps(FileSystemCacheIndex.search)
    def search(self, *args: Any, **kwargs: Any):
        """
        Override for `.search` to follow behaviour of other `FileSystemIndex`'s.

        See `.search_intake` to search the intake catalog.
        """
        return self.filesystem(*args, **kwargs)
