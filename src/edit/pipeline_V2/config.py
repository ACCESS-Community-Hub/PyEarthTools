from typing import Any


class conf():
    """Config class for edit.pipeline_V2"""
    RUN_PARALLEL: bool = True
    DASK_CONFIG: dict[str, Any] = {}

