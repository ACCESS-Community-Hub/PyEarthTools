# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import importlib
from pathlib import Path
import yaml
import warnings

from edit.training.models.utils import get_callable

default_networks = Path(__file__).parent / "networks.yaml"


class Networks:
    def __init__(self, path: Path = None):
        path = path or default_networks

        self._networks: dict
        self._networks = yaml.safe_load(open(path))
        self._imported_networks = []
        self._import_networks()

    def _import_networks(self, specific_key=None):
        networks = self._networks
        if specific_key:
            networks = {specific_key: networks[specific_key]}

        for key, value in networks.items():
            if value in self._imported_networks:
                continue
            try:
                setattr(self, value, get_callable(key))
                self._imported_networks.append(value)
            except AttributeError as e:
                if "partially initialized" in e.__str__():
                    pass
                else:
                    raise e
            except Exception as e:
                warnings.warn(f"Unable to import {key!r} due to {e!r}", ImportWarning)

    @property
    def __dict__(self):
        importlib.invalidate_caches()
        self._import_networks()
        return {key: getattr(self, key) for key in self._imported_networks}

    def __getattr__(self, key):
        if key not in self._networks.values():
            raise AttributeError(f"module 'edit.training.models.networks' has no attribute {key!r}")

        importlib.invalidate_caches()
        # warnings.resetwarnings()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ImportWarning)
            search_key = {v: k for k, v in self._networks.items()}[key]
            self._import_networks(specific_key=search_key)

        if key not in self._imported_networks:
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        else:
            return getattr(self, key)
