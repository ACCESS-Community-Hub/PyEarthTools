from contextlib import contextmanager


# default scheduler string to set "single-threaded" mode.
_STR_DASK_SYNC_SCHEDULER = "synchronous"


@contextmanager
def _set_synchronous_dask():
    """
    Wrapper to set `dask` to single-threaded mode. Note: "single-threaded" in `dask`-land
    (specifically) is the same as "synchronous".

    This handles the case where dask is _not_ installed. In which case it does a pass-through.

    IMPORTANT: never nest this context manager or call dask.config.reset() or attempt to update any
    configs inside this context. Doing so may invalidate the "synchronous" setting.

    Example:
        def do_stuff(...):
            # I can now (optionally) fork other processes here - without confusing dask.
            # IMPORTANT: I shouldn't try to reintroduce parallelism using dask here
            ...

        with _set_synchronous_dask():
            do_stuff(...)
    """
    try:
        # this import order is important for the "distributed" configs to be recognized
        import dask
        import dask.config

        # NOTE: if you don't have dask.distributed, this setting may not work as intended.
        # so you will have to manually deal with it in the compute level.
        import dask.distributed

        # set state to desired config
        with dask.config.set(scheduler=_STR_DASK_SYNC_SCHEDULER):
            yield
    except ImportError:
        yield
