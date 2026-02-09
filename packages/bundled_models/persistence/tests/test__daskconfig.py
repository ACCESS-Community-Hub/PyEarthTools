"""
Tests that dask is actually in synchronous/signle-threaded mode
"""

from dataclasses import dataclass
import numpy as np
import persistence as pet_persist
import persistence.daskconfig as pet_daskconfig


@dataclass
class _PyTestThreadInfo:
    id_thread_kern: int  # usually same as process id
    id_thread_py: int  # python read id
    id_process: int  # process id for current worker
    num_cpus: int  # number of cpus


def _fn_dask_get_thread_info(count):
    return _make_thread_info()


def _cmp_thread_info(
    thread_info_a: _PyTestThreadInfo, thread_info_b: _PyTestThreadInfo
) -> int:
    """
    Works like strcmp, thread info is the same => return 0, otherwise they are different.
    """
    # Each critera will return 0 if they are equal or 1 if they are not. A larger number implies
    # that there is larger discrepency.
    # NOTE: cpu checks is not strictly required, but helpful to know, since it is not an expected
    # scenario unless running multi-node.
    count_diff = (
        int(thread_info_a.id_thread_kern != thread_info_b.id_thread_kern)
        + int(thread_info_a.id_thread_py != thread_info_b.id_thread_py)
        + int(thread_info_a.id_process != thread_info_b.id_process)
        + int(thread_info_a.num_cpus != thread_info_b.num_cpus)
    )
    return count_diff


def _is_multithreaded_compute(list_thread_info) -> bool:
    """
    Returns true if the list of thread_info have different threads or processes.
    """
    ref_thread_info = list_thread_info[0]
    flag_has_different_threads = False
    for i, v in enumerate(list_thread_info):
        # ignore reference (i == 0) and update flag if a difference is spotted
        if i != 0 and _cmp_thread_info(v, ref_thread_info) != 0:
            flag_has_different_threads = True
            break
    return flag_has_different_threads


def _make_thread_info():
    """
    Creates the current thread info for the given context. This shouldn't be a fixture, it needs to
    be called internally by a worker in the test.
    """
    import threading
    import os

    obj_thread_py: threading.Thread = threading.current_thread()
    return _PyTestThreadInfo(
        id_thread_kern=obj_thread_py.native_id,
        id_thread_py=obj_thread_py.ident,
        id_process=os.getpid(),
        num_cpus=os.cpu_count(),
    )


def test_dask_single_threaded():
    """
    Set single threaded mode and check that the thread ids are the same for each worker.
    """
    import dask
    import dask.config
    import dask.distributed
    import dask.dataframe as _dd
    import dask.array as _da

    main_thread_info: _PyTestThreadInfo = _make_thread_info()

    # we still set multiprocess here to check if our context manager is working as expected.
    dask.config.config["scheduler"] = "processes"
    dask.config.refresh()

    # partition task of processing 100 items by number of ccpus
    _chunks = (min(main_thread_info.num_cpus, 100),)
    _dask_df = _dd.io.from_dask_array(
        _da.from_array(np.arange(100), chunks=_chunks),
        columns=["x"],
    )

    # run computation in context manager
    with pet_daskconfig._set_synchronous_dask():
        results = _dask_df.apply(
            _fn_dask_get_thread_info, axis=1, meta=(None, "object")
        ).compute()
        assert not _is_multithreaded_compute(results)


def test_dask_default_multithreaded():
    """
    Tests dask without singlethreaded context management.
    """
    # NOTE: this namespacing does not guarentee dask is out of scope in other tests
    import dask
    import dask.config
    import dask.distributed
    import dask.dataframe as _dd
    import dask.array as _da

    # intentionally set to multiprocess mode (which is usually the case with e.g. xarray)

    main_thread_info: _PyTestThreadInfo = _make_thread_info()
    dask.config.config["scheduler"] = "processes"
    dask.config.refresh()

    # partition task of processing 100 items by number of ccpus
    _chunks = (min(main_thread_info.num_cpus, 100),)
    _dask_df = _dd.io.from_dask_array(
        _da.from_array(np.arange(100), chunks=_chunks),
        columns=["x"],
    )
    # get results
    results = _dask_df.apply(
        _fn_dask_get_thread_info, axis=1, meta=(None, "object")
    ).compute()

    # --- check if there are sufficient threads on system
    if len(results) <= 1:
        print("Insufficient cores/threads to do multi-process tests")
        return

    assert _is_multithreaded_compute(results)
