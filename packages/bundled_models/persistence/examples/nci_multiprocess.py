"""
This example is a WORK IN PROGRESS.

Use multiprocessing to delegate chunks to various processes, in order to compute the persistence
method in an embarassingly parallel fashion.

This run specifically targets satellite data on NCI, but is also the most common usecase for
parallel compute.

TODO:
    [ ] example with PersistenceRM wrapper
    [ ] example with plain persistence execution on a separate spawned process (separate GIL)

NOTE:
    These examples will eventually also go into the global tutorials.
"""

def persistence_pipeline_library():
    # TODO: use library directly under main guard
    pass

def persistence_pipeline_spawnproc():
    # TODO: spawn process as separate python command
    pass


if __name__ == "__main__":
    pass
