from typing import Any, Literal


class conf:
    """Config class for edit.pipeline_V2"""

    RUN_PARALLEL: bool = True

    START_DASK: bool = True
    DASK_CONFIG: dict[str, Any] = {} #"processes": False
    DEFAULT_INTERFACE: Literal['Futures', 'Delayed', 'Serial'] = 'Delayed'

    MAX_FILTER_EXCEPTIONS: int = 10
    MAX_ITERATOR_EXCEPTIONS: int = 20
    DEFAULT_IGNORE_EXCEPTIONS: tuple[Exception, ...] = tuple()
