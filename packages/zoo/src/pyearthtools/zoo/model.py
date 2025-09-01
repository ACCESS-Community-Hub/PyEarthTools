# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: F821

"""Base model class"""


from __future__ import annotations

from abc import abstractmethod
import functools
import itertools
from typing import Any, Optional, Literal
from pathlib import Path


import os
import shutil
import time
import logging
import warnings

from multiurl import download

from pyearthtools.zoo.utils import create_mapping, split_name_assignment

LOG = logging.getLogger("pyearthtools.zoo")

REPR_STRING = """`pyearthtools.zoo` Forecast model

Model Name:          {model!r}
Data Source:         {config!r}
Output Directory:    {output!r}

{kwargs}
"""


class Timer:
    """
    Timer

    Record and log the execution time of code within this context manager.
    """

    def __init__(self, title: str, logger: logging.Logger | None = None):
        self.title = title
        self.logger = logger
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print("%s: took %.2f seconds.", self.title, elapsed)
        # log = self.logger or LOG  # TODO: Bring this back
        # log.debug("%s: took %.2f seconds.", self.title, elapsed)


class BaseForecastModel:
    """
    Setup `BaseForecastModel`

    **A child must at least implement the `.load` function to pass back a `pyearthtools.training.wrapper.Predictor`.**

    ## Setup
        - Setting `_default_config_path` provides a default config path.
            This should be given, otherwise it must be set by the user each time.
        - Setting `_times` allows a model to specify which time deltas need to be
            retrieved for predictions. Used for live download.
        - Setting `_download_paths` specifies files to download.
        - Setting `_name` provides the name of the model. It is best to set
            this identical to where the model is registered to.
            If not given, will be the class name. Use '/' to set categories.

    ### `_download_paths`
        Setting `_download_paths` in the class will allow those assets to be automatically retrieved and stored.
        They are then accessible underneath a directory retrievable from `self.assets`.

        If given as a str the last '/' will be used as the name, or if given as a tuple,
        the first element is the link, and the second the name.

        These paths can be to either a file or a zip file on a server or on the local machine.

        If the assets should be downloaded each time, set `_redownload_each_time` to True.


    ## Config Folder
    The config folder for a model can use the following conventions to ease in setup

    `Data/` - Location for all data loaders
    `Pipeline/` - Location for all pipelines

    It is assumed that most data configs will have a pipeline name identically to them for
    loading and preparing the data, however, the following exception applies.
    If a `Data` config has a `()`, with a str inside, it represents a different data source, but the same pipeline,
    this is can be useful for setting by different sources of the same data, link for downloading, archived data or experiments.

    Additionally, any data with a `-` represents an ancillary source, i.e. forcings and will not be included
    in the available data sources. Any text prior to the `-` represents the parent source and any after is it's purpose.

    Getting `ancillary_pipeline` will give back a dictionary of ancillary pipelines associated with the chosen source.

    When creating a model subclass set `_default_config_path` to the default path.

    A user can provide a `config_path` during `__init__` to allow access to user defined configs.
    This allows experiments to be easily run, and will follow the conventions outlined above.
    i.e. Providing a data config with `()` will use the base pipeline.

    ### Examples
    Consider the following structure.

    >>> ├── Data
    >>> │   ├── ERA5-Forcings.yaml
    >>> │   ├── ERA5(cds)-Forcings.yaml
    >>> │   ├── ERA5(cds).yaml
    >>> │   └── ERA5.yaml
    >>> └── Pipeline
    >>>     └── ERA5.yaml

    A user can request either `ERA5` or `ERA5(cds)` as the data source, these two sources are then loaded and use
    `Pipeline/ERA5.yaml` as it's pipeline.

    When getting `ancillary_pipeline`, either `ERA5-Forcings` or `ERA5(cds)-Forcings` will be used, dependent on
    the data source as detailed above.
    If a `Pipeline/ERA5-Forcings.yaml` existed, both sources would then use this as their pipeline.

    ### Configurable Config Path
    Using `pyearthtools.config` the paths in which config files for `pyearthtools.zoo` can be adjusted.
    This can be done by either setting `configs` in `~/.config/pyearthtools/models.yaml` or setting `pyearthtools_MODELS__CONFIGS`
    in the environment.

    An environment can define a list of paths split by `:` at `pyearthtools_MODELS__CONFIGS`. These will be added to the
    valid pipelines, with the model class name added to the end.

    For most models this should be the full categorical path of the model, see each model for it's `_name`.
    If not set will be the class name.


    ### Config Assignments
    Specifying a '{}' after a config selection allows a user to specify replacement keys for the pipeline.

    All keys in a pipeline need to be surrounded by '__', so that a key `ID` corresponds in the config to `__ID__`,
    Say the `ERA5` pipeline contains a key: '__ID__', to allow a user to select a certain ID at the time of running,
    the config can be specified as:

    ```
    'ERA5{ID=42}'
    ```

    This will replace `__ID__` inside the config before it is loaded with '42'.
    The replacement value will be a str.

    #### Default Assignments
    | Key | Value |
    | --- | ----- |
    | pyearthtools_ASSETS | Asset path to this model |
    | pyearthtools_MODELS_DEFAULT_CONFIG | Default config path for a model |
    | FILE | Folder containing the loaded config |
    | OUTPUT_DIR | Output location as specified by the user |

    If ':' follows the KEY part and still within '__*__', anything following will be considered the default value.

    #### Class assignments
    Assignments like shown above can also be provided within `self._default_assignments` which will be used when loading a
    `Pipeline`.

    ```
    self._default_assignments = {ID = 42}
    ```


    ## Assets
    Assets will be saved at the location given in the config at `models.assets`.
    This can be cnanged by either setting `assets` in `~/.config/pyearthtools/models.yaml` or setting `pyearthtools_MODELS__ASSETS`
    in the environment.

    The model name is appended to this path, as specify only the overall `pyearthtools` asset path.

    This asset path is then accessible from `self.assets`.

    ## Caching Inputs
    Setting `config_path` allows for the inputs of a model to be cached out before inference.
    This may be especially useful for sanity checking, or preloading downloaded data before switching to a compute node.

    `data_cleanup` defines how to manage this cache, by default, will remove any data over 1 day old,
    and limit the directory size to `10GB`, see `pyearthtools.data.indexes.CachingIndex` for more information.

    The model `_name` and pipeline name is automatically added to the path to prevent collisions. So:

    If `config_path` is `/data/goes/here`, and the model is `Model/Name`, with pipeline `PipelineName`

    The full path is `/data/goes/here/Model/Name/PipelineName`

    The pattern of the cache will then take over.

    """

    _times: list[int] = [0]  # Times to get
    _download_paths: list[str] = []  # Download urls to get
    _redownload_each_time: bool = False
    _name: str | None = None

    _default_assignments: dict[str, Any] = {}  # Class attribute assignments

    def __init__(  # pylint: disable=R0913
        self,
        *,
        pipeline_name: Optional[str] = None,
        pipeline=None,
        output: Optional[os.PathLike] = None,
        config_path: Optional[os.PathLike] = None,
        data_cache: Optional[os.PathLike] = None,
        data_cleanup: dict[str, Any] | str | None = None,
        delete_cache: Optional[bool] = False,
        download_assets: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        BaseForecastModel

        Must be implemented by a child class to setup a model

        A child must at least implement the `.load` function to pass back a `pyearthtools.training.wrapper.Predictor` wrapper.

        Args:
            pipeline_name: Pipeline name to use, must be in `valid_pipeline`
            pipeline: Already-loaded pipeline object (alternative to the pipeline name)
            output Location to save predictions
            config_path: Override for config path to find Data & Pipelines. Defaults to None.
            data_cache: Location to set a data cache for, automatically adds model name & pipeline to path. Defaults to None.
            data_cleanup: Config for cleanup for data_cache. Defaults to None.
            delete_cache: Delete all data in cache. Defaults to False.
            download_assets (bool, optional):
                Whether to download assets.
                Will be called anyway upon first call to `.index`
                Defaults to False.
            **kwargs (Any, optional):
                All extra kwargs used when getting the DataIndex.

        Raises:
            ValueError:
                If `pipeline` not in `._valid_pipeline()` and a valid loaded pipeline is not supplied
        """

        if download_assets:
            with self.timer("Downloading assets"):
                self.download_assets()  # Download assets

        self._config_path = config_path
        logger = self.log()
        import pyearthtools.zoo  # pylint: disable=C0321

        if pipeline is not None and pipeline_name is not None:
            raise ValueError("Cannot initialise with both a named pipeline and an im-memory pipeline")

        if pipeline is None and pipeline_name is None:
            raise ValueError("Cannot initialise, require either a named pipeline or an in-memory pipeline")

        # Using an in-memory pipeline
        if pipeline is not None:
            self._pipeline = pipeline
            self._pipeline_name = "User-supplied pipeline"

        # Establishing pipeline from configuration
        else:

            # Sort out data access as specified in the pipeline
            if any(map(lambda x: x in pipeline_name.lower(), pyearthtools.zoo.LIVE_SUBSTRINGS)) and data_cache is None:
                ## Must setup a cache for live data
                data_cache = self.get_config("cache")

            # Validate the pipeline file
            if not self.is_valid_pipeline(pipeline_name, config_path=self._config_path):
                raise ValueError(
                    f"Cannot find config: {pipeline_name} in {config_path}\n. "
                    f"Valid items: {list(self._valid_pipeline(config_path=self._config_path).keys())}"
                )

            # Validation and init passed, log success and carry on
            self._pipeline_name = pipeline_name
            self._pipeline = None
            message = f"Using pipeline: {self._pipeline_name}"
            logger.debug(message)

        self.output = output
        self._kwargs = kwargs
        self._data_cleanup = data_cleanup

        import pyearthtools.data  # pylint: disable=C0321

        self._data_cache = pyearthtools.data.utils.parse_path(data_cache) if data_cache is not None else data_cache
        self._delete_cache = delete_cache

    @classmethod
    def get_name(cls) -> str:
        """
        Get name of this class.

        Can be overriden by setting `_name`, if not given, will be `cls.__name__`.
        """
        return str(getattr(cls, "_name", None) or cls.__name__)

    @classmethod
    # @property
    # @functools.cache
    def log(cls) -> logging.Logger:
        """Model specific logger"""
        # model_specific = logging.getLogger(f"pyearthtools.zoo.{self.get_name()}")
        # tuple(model_specific.addHandler(handler) for handler in LOG.handlers)
        return LOG.getChild(cls.get_name())

    @classmethod
    def get_config(cls, key: str, default: Any = None) -> Any:
        """Get config for `key` from `pyearthtools.config`"""
        import pyearthtools.utils

        return pyearthtools.utils.config.get(f"models.{key}", default=default)

    def timer(self, title: str) -> Timer:
        """
        Get timer context local to this object.

        Args:
            title (str):
                Name of timer

        Returns:
            (Timer):
                Timer context
        """
        return Timer(title, logger=self.log)

    @functools.cached_property
    def assets(self) -> Path:
        """
        Get assets directory.
        Set in config by `models.assets`, therefore can be configured by the user in `~/.config/pyearthtools/models.yaml`,
        or by setting `pyearthtools_MODELS__ASSETS` in the environment.
        """
        import pyearthtools.data

        asset_dir = pyearthtools.data.utils.parse_path(Path(self.get_config("assets")).expanduser().resolve())
        return asset_dir / self.get_name()

    @functools.cached_property
    def cache(self) -> Path | None:
        """
        Get cache directory
        """
        return (
            (Path(self._data_cache) / self.get_name()).expanduser().resolve()
            if isinstance(self._data_cache, (Path, str))
            else self._data_cache
        )

    def download_assets(self) -> None:
        """Download all assets in `_download_paths`, and store in `.assets`"""
        for file in self._download_paths:
            if isinstance(file, tuple) and len(file) == 2:
                link = file[0]  # Link as first element
                name = file[1]  # Name as second element
            elif isinstance(file, str):
                name = str(file).rsplit("/", maxsplit=1)[-1]  # Get file name as last element of url
                link = file
            else:
                raise TypeError(f"Cannot parse file of type {type(file)}.")

            asset = Path(self.assets) / name

            if not asset.exists() or self._redownload_each_time:
                try:
                    os.remove(str(asset))
                except (OSError, FileNotFoundError, PermissionError):
                    pass

                asset.parent.mkdir(exist_ok=True, parents=True)
                # self.log.info("Retrieving %s", link)

                if Path(link).exists():
                    # self.log.debug(f"Copying {link} to {asset}.")
                    shutil.copyfile(link, str(asset) + ".download")
                else:
                    download(link, str(asset) + ".download")

                os.rename(str(asset) + ".download", asset)

                if str(file).endswith(".zip") and getattr(self, "_allow_zip", False):
                    self.log.info("Unpacking %s", asset)
                    shutil.unpack_archive(asset, extract_dir=asset.parent)

    @classmethod
    @functools.cache
    def get_all_config_paths(cls, config_path: os.PathLike | None) -> tuple[Path, ...]:
        """
        Get all config paths associated with this model.

        Args:
            config_path: Defined Config path to add.

        Returns:
            (tuple[Path, ...]):
                All config paths

        Raises:
            ValueError:
                If no config paths found.
        """
        paths = []

        if config_path is not None:
            paths.append(Path(config_path))

        env_path = cls.get_config("configs")

        if env_path is not None:
            if isinstance(env_path, str):
                env_path = env_path.split(":")

            for environ_path in env_path:
                if Path(environ_path).exists():
                    paths.append(Path(environ_path) / cls.get_name())

        if len(paths) == 0:
            raise ValueError("No config paths could be established.")
        cls.log().debug(f"Config paths: {paths}")
        return tuple(paths)

    @classmethod
    def valid_pipelines(cls, ancillary: bool = False, *, config_path: os.PathLike | None = None):
        """
        Get valid pipeline list at `config_path`.

        See `_valid_pipeline` for full docs.
        """
        return list(cls._valid_pipeline(ancillary=ancillary, config_path=config_path).keys())

    @classmethod
    @functools.cache
    def _valid_pipeline(
        cls, ancillary: bool = False, *, config_path: os.PathLike | None = None
    ) -> dict[str, str | None]:
        """
        Get valid pipeline configs.

        Setting `config_path` allows checking of user defined directories.

        Args:
            ancillary: Whether to get only ancillary pipelines. Defaults to False.
            config_path: Allows a user to find all valid configurations.

        Returns:
            (dict[str, str | None]):
                Data name and associated pipeline

        Raises:
            ValueError:
                If no config paths found.
        """

        config_paths = cls.get_all_config_paths(config_path)

        def find_elements(sub_path: str, suffix=[".yaml", ".pipe", ".pipeline"]) -> list[str]:
            """
            Get all configs in path, subsetting by suffix

            If ancillary set, get ancillary files, else get not ancillary
            """

            def find_paths(path: Path) -> list[str]:
                return [
                    p.with_suffix("").name
                    for p in (Path(path) / sub_path).glob("*.*")
                    if p.suffix in ([suffix] if not isinstance(suffix, list) else suffix)
                ]

            paths: list[str] = list(itertools.chain(*(find_paths(config_path) for config_path in config_paths)))

            cls.log().debug(f"Pipeline paths for {sub_path!r}: {paths}")

            def valid(x: str) -> bool:
                return "-" in str(x) if ancillary else "-" not in str(x)

            paths: list[str] = [p for p in paths if valid(p)]
            return paths

        data_elements = find_elements("Data")  # Find configs from the Data directory
        pipeline_elements = find_elements("Pipeline")  # Find configs from the Pipelines directory
        mappings = create_mapping(data_elements, pipeline_elements)
        return mappings

    @classmethod
    def is_valid_pipeline(cls, pipeline_name: str, config_path: os.PathLike | None = None) -> bool:
        """
        Check if `pipeline` is a valid pipeline

        Args:
            pipeline: Pipeline name to check if valid
            config_path: Path to search for configuration

        Returns:
            (bool):
                If `pipeline` is valid.
        """
        valid_pipelines = cls._valid_pipeline(config_path=config_path)
        pipeline, _ = split_name_assignment(pipeline_name)
        found_it = pipeline in valid_pipelines

        if found_it:
            print(f"Valid {pipeline_name} found in path {config_path}")
        else:
            print(f"Could not find/validate {pipeline_name} in path {config_path}")
        return found_it

    @functools.cache  # pylint: disable=W1518
    def _get_cache(self, cache: os.PathLike):
        """
        CachingIndex for temp data storage.

        Automatically setup with cleanup of `delta = 1 day, dir_size = '10 GB'`
        If `data_cleanup` given on `init` use that instead.

        If `delete_cache` given, delete all data underneath.

        Args:
            cache (os.PathLike):
                Location to make cache for
        """
        import pyearthtools.pipeline

        if self._delete_cache and cache is not None:
            from pyearthtools.data.indexes.utilities import delete_files

            self.log.debug("Deleting all data in cache at %r", cache)
            delete_files.delete_path(cache)

        return pyearthtools.pipeline.modifications.Cache(  # pylint: disable=E0110,E1120
            cache=Path(cache),
            cache_validity="deleteF",
            cleanup=self._data_cleanup or {"delta": 1, "dir_size": "10 GB"},
            pattern_kwargs={"directory_resolution": "year"},
        )

    def load_pipeline(self, pipeline: str, data: bool = True, ancillary: Optional[str] = None, **kwargs: Any) -> "pyearthtools.pipeline.Pipeline":  # type: ignore
        """

        Hook to allow modification of how `pipeline` is loaded.

        Args:
            pipeline: Path to pipeline file to open.
            data: If pipeline is the data source or pipeline.
            ancillary: Name of ancillary pipeline if ancillary pipeline.
            kwargs: Assignments to pass to `pyearthtools.pipeline.load`

        Returns: Loaded pipeline

        Usage:
            A child model could override this to assign values within `__KEY__` keys inside the `Pipeline`.
            Or add a step.
        """
        import pyearthtools.pipeline

        return pyearthtools.pipeline.load(
            pipeline,
            **kwargs,
        )

    def _get_pipeline(
        self,
        pipeline_name: str,
        cache: os.PathLike | None = None,
        ancillary: Optional[str] = None,
    ) -> "pyearthtools.pipeline.Pipeline":  # type: ignore  # noqa: F821
        """
        Load pipeline from specified `pipeline_name`

        Uses valid pipeline dictionary, so will ignore elements in () when finding the associated pipeline.

        Will either not or only retrieve from ancillary sources depending on `ancillary`

        Args:
            config (str):
                Config name to get data and pipeline of
            cache (os.PathLike | None, optional):
                Path to add a CacheIndex for after data source. Defaults to None.
            ancillary (Optional[str], optional):
                Name of ancillary pipeline being retrieved if given. Otherwise
                do not allow ancillary pipelines to be loaded. Defaults to None.

        Raises:
            RuntimeError:
                If `_config_path` is not set, must be set by child class

        Returns:
            (pyearthtools.pipeline.Pipeline):
                Loaded Pipeline
        """
        import pyearthtools.pipeline

        config_paths = self.get_all_config_paths(self._config_path)

        assignment = dict(self._default_assignments or {})

        pipeline_name, name_assignment = split_name_assignment(pipeline_name)
        assignment.update(name_assignment or {})

        def load(sub_path: Literal["Data", "Pipeline"], item: str) -> "pyearthtools.pipeline.Pipeline":  # type: ignore
            """Load config"""

            def file_at_path(path: os.PathLike):
                return tuple((Path(path) / sub_path).glob(f"{item}.*"))

            # Get files from both config_path and default if different
            paths: list = list(itertools.chain(*(file_at_path(config_path) for config_path in config_paths)))

            file = list(set(paths))

            if len(file) > 1:
                raise ValueError(f"Found multiple files under {sub_path}/{item}: {file}. Cannot pick one.")

            return self.load_pipeline(
                file[0],
                data=sub_path == "Data",
                ancillary=ancillary,
                **(assignment if assignment else {}),
                pyearthtools_ASSETS=self.assets,
                pyearthtools_MODELS_DEFAULT_CONFIG=self._config_path,
                OUTPUT_DIR=self.output,
            )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pyearthtools.pipeline.PipelineWarning)
            valid = self._valid_pipeline(
                ancillary=ancillary is not None, config_path=self._config_path
            )  # Get valid sources

            pipeline = None
            if valid[pipeline_name] is not None:
                pipeline = load("Pipeline", valid[pipeline_name])  # type: ignore # Load pipeline source

            data = load("Data", pipeline_name)  # Load data source

            if cache is not None:  # Add cache if given
                data = data + self._get_cache(Path(cache) / pipeline_name)

            pipeline = data + pipeline if pipeline is not None else data  # Combine the two pipelines together

        return pipeline

    @functools.cached_property
    def pipeline(self) -> "pyearthtools.pipeline.Pipeline":  # type: ignore  # noqa: F821
        """
        Get pipeline as configured in the init.
        """

        if self._pipeline:
            return self._pipeline

        logger = self.log()

        pipe = self._get_pipeline(self._pipeline_name, cache=self.cache)
        logger.debug("Using Pipeline: %r", pipe)
        return pipe

    @functools.cached_property
    def ancillary_pipeline(self) -> dict[str, "pyearthtools.pipeline.Pipeline"]:  # type: ignore  # noqa: F821
        """
        Ancillary Pipelines

        Get all ancillary pipelines associated with the selected one.

        Ancillaries are marked with a '-' with the prior representing the core, and the post the name of the ancillary.

        Returns:
            (dict[str, pyearthtools.pipeline.Pipeline]):
                Name of ancillary: Loaded Pipeline

        """
        valid = self._valid_pipeline(ancillary=True, config_path=self._config_path)  # Get valid ancillary pipelines

        def find_ancillary(name: str):
            """
            Get ancillaries connected to the specified pipeline
            """

            ancillary: dict = {}
            for config in [v for v in valid if name == v.split("-")[0]]:
                ancillary[config.split("-")[-1]] = self._get_pipeline(
                    config,
                    cache=self.cache,
                    ancillary=name,
                )
            return ancillary

        import re

        ancillary = find_ancillary(self._pipeline_name)  # Get with '()'
        if len(ancillary) == 0:
            ancillary = find_ancillary(re.sub(r"\(.*\)", "", self._pipeline_name))

        return ancillary

    def data(self, basetime: str) -> list[Any]:
        """
        Get data from pipeline

        Used to download for live runs

        Args:
            basetime (str):
                Time that a prediction would be run at
        """
        import pyearthtools.data
        import tqdm.auto as tqdm

        pipeline = self.pipeline
        # if "CachingIndex" in self.pipeline.steps:
        #     pipeline = self.pipeline.step("CachingIndex")

        data = []
        with self.timer("Data retrieval"):
            for t in tqdm.tqdm(self._times, desc="Getting data"):
                data_time = pyearthtools.data.pyearthtoolsDatetime(basetime) + t

                d = pipeline[data_time]
                for key, _ in self.ancillary_pipeline.items():
                    try:
                        _ = pipeline[data_time]  # type: ignore
                    except Exception as e:  # pylint: disable=W0718
                        self.log.debug("An error occured when getting data from ancillary %s: %s.", key, e)
                data.append(d)
        self.log.debug("Loaded all data.")
        return data

    @abstractmethod
    def load(self, *args, **kwargs) -> tuple["pyearthtools.training.wrapper.Predictor", dict[str, Any]]:  # type: ignore
        """
        Load `pyearthtools.training.wrapper.Predictor`, and provide kwargs for `pyearthtools.training.MLDataIndex`.

        Must accept user passed kwargs.
        """
        raise NotImplementedError("Child class must provide a load method.")

    def _get_index(self, cache: os.PathLike | None, **kwargs) -> "pyearthtools.training.MLDataIndex":  # type: ignore
        """
        Get `DataIndex` of model

        Args:
            cache (os.PathLike):
                Location to set Index cache to
            **kwargs (Any, optional):
                Kwargs to pass to load function of model

        Returns:
            (pyearthtools.training.trainer.MLDataIndex):
                Index of model
        """
        import pyearthtools.training
        import pyearthtools.data
        from pyearthtools.zoo import __version__ as VERSION

        kwargs["weights_only"] = (
            False  # TODO: Remove this, insecure if checkpoint is untrusted, weights_only is preferred
        )

        model, index_kwargs = self.load(**kwargs)  # Load model and trainer

        # self.log.debug(f"Loading returned {model.__class__ =}")
        # self.log.debug(f"Loading returned {index_kwargs =}")

        post_transforms = index_kwargs.pop("post_transforms", pyearthtools.data.TransformCollection())

        full_index_kwargs = dict(self.get_config("training_data_index"))
        full_index_kwargs.update(dict(index_kwargs))

        default_pattern_kwargs = self.get_config("pattern", {})

        if full_index_kwargs["pattern"] in default_pattern_kwargs:
            pattern_kwargs = default_pattern_kwargs[full_index_kwargs["pattern"]]
            pattern_kwargs.update(full_index_kwargs.pop("pattern_kwargs", {}))
            full_index_kwargs["pattern_kwargs"] = pattern_kwargs

        post_transforms = (
            pyearthtools.data.transforms.attributes.set_attributes(
                pyearthtools_models=f"{self.get_name()}: {VERSION}",
                apply_on="both",
            )
            + post_transforms
        )
        # self.log.debug(f"Initialising MLDataIndex with {full_index_kwargs =}")

        return pyearthtools.training.MLDataIndex(
            model,
            cache=cache,
            post_transforms=post_transforms,
            data_interval=full_index_kwargs.pop("data_interval", None),
            **dict(full_index_kwargs),
        )  # Setup index

    @functools.cached_property
    def index(self) -> "pyearthtools.training.MLDataIndex":  # type: ignore
        """Get pipeline as an `MLDataIndex`"""
        with self.timer("Downloading assets"):
            self.download_assets()  # Download assets

        return self._get_index(self.output, **self._kwargs)

    @functools.cached_property
    def model(self) -> "pyearthtools.training.PredictionWrapper":  # type: ignore
        return self.load()[0]

    def run(self, *args, **kwargs) -> Any:
        """
        Run model

        Using pipeline, and overwritten load function, create a `DataIndex` for the model,
        and run a prediction

        All args, and kwargs passed through

        Raises:
            RuntimeError:
                If a DataNotFoundError occurs

        Returns:
            (Any):
                Result of running the index
        """
        import pyearthtools.data

        excep = None
        try:
            with self.timer("Running Prediction"):
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=pyearthtools.data.pyearthtoolsDataWarning)
                    return self.index(*args, **kwargs)
        except pyearthtools.data.DataNotFoundError as e:
            excep = e
        raise RuntimeError(
            "An `pyearthtools.data.DataNotFoundError` occured. "
            "Try running this command with the `--data` flag to check data availability.\n"
            "If running on a compute node, try downloading the data online."
        ) from excep

    def __call__(self, *args, **kwargs):
        """Run model when called."""
        return self.run(*args, **kwargs)

    def __getitem__(self, idx):
        return self.run(idx)

    def search(self, *args, **kwargs) -> dict[str, Path]:
        """
        Run a safe search on the index, skipping override
        """
        from pyearthtools.utils.context import ChangeValue

        with ChangeValue(self.index, "_override", False):
            return self.index.search(*args, **kwargs)

    def __repr__(self):
        def spacing(k, v):
            return f"{k}:{''.join([' '] * (20 - len(k)))}{v!r}"

        kwargs_str = "\n".join(spacing(key, value) for key, value in self._kwargs.items())
        return REPR_STRING.format(
            model=self.get_name(),
            config=self._pipeline_name,
            output=self.output,
            kwargs=kwargs_str,
        )
