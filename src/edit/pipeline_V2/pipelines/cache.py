# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Any, Optional, Union, Literal
import warnings
import functools

from pathlib import Path
from hashlib import sha512
import shutil

import edit.data
from edit.data.patterns import PatternIndex, PatternTimeIndex

from edit.pipeline_V2.controller import PipelineMod
from edit.pipeline_V2.warnings import PipelineWarning

CACHE_HASH_NAME = ".cache_hash"
# PIPELINE_SAVE_NAME = "pipeline.yaml"


class Cache(PipelineMod):
    _cache: edit.data.indexes.FunctionalCacheIndex

    def __init__(
        self,
        cache: Union[str, Path],
        pattern: Optional[Union[str, PatternIndex]] = None,
        pattern_kwargs: dict[str, Any] = {},
        cache_validity: Literal["trust", "delete", "warn", "keep", "override"] = "warn",
        **kwargs,
    ):
        super().__init__()
        self.record_initialisation()

        self.cache_behaviour = cache_validity
        self._cache = edit.data.indexes.FunctionalCacheIndex(
            cache,
            pattern,
            function=self._generate,
            pattern_kwargs=pattern_kwargs,
            **kwargs,
        )

    def _generate(self, idx):
        return super().__getitem__(idx)

    def __getitem__(self, idx):
        return self._cache[idx]

    @property
    def cache(self) -> edit.data.indexes.FunctionalCacheIndex:
        return self._cache

    """
    Hashing and pipeline saving
    """

    @property
    def cache_hash_file(self) -> Path:
        """Get the hash file name"""
        return Path(self.cache) / CACHE_HASH_NAME

    @property
    def pipeline_save_file(self) -> Path:
        """Get the pipeline save file name"""
        return Path(self.cache) / PIPELINE_SAVE_NAME

    def make_cache_hash(self):
        """
        Attempt to make cache hash, if fails do nothing and try again later.
        """
        if self.cache_hash_made:
            return
        try:
            self.cache_validity()
        except Exception as e:
            warnings.warn(f"Cache hash could not be made yet. \n{e}", PipelineWarning)
        self.cache_hash_made = True

    def make_pipeline_save(self):
        """
        Attempt to make pipeline file, if fails do nothing and try again later.
        """
        if self.pipeline_save_made:
            return

        try:
            self.save(self.pipeline_save_file)
        except Exception as e:
            warnings.warn(f"Pipeline file could not be made yet. \n{e}", PipelineWarning)
        self.pipeline_save_made = True

    def cache_validity(self) -> bool:
        """
        Check the cache validity according to `cache_validity` passed in `__init__`.
        """
        if not self.cache_hash_file.exists():
            self._save_hash()
            return True

        if not self._get_saved_cache():
            self._save_hash()
            self.pipeline_save_made = False
            self.make_pipeline_save()
            return False

        cache_validity = self.hash == self._get_saved_cache()

        if cache_validity or self.cache is None:
            return True

        if self.cache_behaviour == "trust":
            self._save_hash()
        elif self.cache_behaviour == "override":
            self._save_hash()
        elif self.cache_behaviour == "keep":
            raise PipelineException(
                "The saved cache hash is not equal to the current hash.\n"
                "Data may be incorrect. If this data can be trusted, change "
                "'cache_validity' to 'trust' or 'warn', or if it needs to be deleted, "
                "set to 'delete', or 'override'."
                f"\nAt location {str(self.cache)!r}"
            )
        elif self.cache_behaviour == "warn":
            warnings.warn(
                "The saved hash and current hash are not the same. "
                "Therefore, data loaded from the cache may not be what is expected.\n"
                "If this cache is valid, pass 'cache_validity' = 'trust' once, to trust this cache.\n"
                "If not, pass 'cache_validity' = 'delete' or 'override', to delete the cache "
                "or override it respectively."
                f"\nAt location {str(self.cache)!r}",
                PipelineWarning,
            )

        elif "delete" in self.cache_behaviour:
            if not "F" in self.cache_behaviour:
                if not input("Cache was invalid, Are you sure you want to delete all cached data? (YES/NO): ") == "YES":
                    warnings.warn(f"Skipping delete.", UserWarning)
                    return False

            warnings.warn(f"Deleting all data underneath '{self.cache}'.", UserWarning)
            shutil.rmtree(self.cache)
            self._save_hash()
        else:
            raise ValueError(f"Cannot parse 'cache_validity' of {self.cache_behaviour}")
        return True

    def _save_hash(self):
        """Save the hash"""
        if not self.cache_hash_file.parent.exists():
            self.cache_hash_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.cache_hash_file, "w") as file:
            file.write(self.hash)

    def _get_saved_cache(self):
        """Get the saved hash"""
        with open(self.cache_hash_file, "r") as file:
            return file.read()

    def _get_saved_pipeline(self):
        """Get the saved pipeline as txt"""
        with open(self.pipeline_save_file, "r") as file:
            return file.read()

    @property
    def hash(self) -> str:
        """
        Get sha512 hash of underlying index
        """
        if isinstance(self.index, DataStep):
            conf: dict[str, dict] = {
                f"{step.__class__}-{num}": step._info_ for num, step in enumerate(self.index._get_steps_for_repr_())
            }
        elif isinstance(self.index, (Catalog, CatalogEntry)):
            conf = self.index.to_dict()
        elif isinstance(self.index, Index):
            conf = self.index.catalog.to_dict()
        else:
            raise ValueError(
                f"Could not convert `index` {type(self.index)!r} into configuration dictionary to make hash of."
            )

        def order_dicts(dictionary: Any) -> Any:
            if not isinstance(dictionary, dict):
                return dictionary

            sorted_keys = list(dictionary.keys())
            sorted_keys.sort()
            for key in sorted_keys:
                dictionary[key] = order_dicts(dictionary[key])
            return dictionary

        conf["CacheConfig"] = self.catalog.to_dict()

        conf = order_dicts(conf)
        configuration = tuple((f"{key}:{value}" for key, value in conf.items()))
        return sha512(bytes(str(configuration), "utf-8")).hexdigest()
