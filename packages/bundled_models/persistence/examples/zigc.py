"""
This example is a WORK IN PROGRESS.

Uses zig to process chunks instead of numpy. Optionally uses multiprocessing to delegate chunks.

SETUP:
    - run setup_dev.sh or appropriate pixi command (TODO)
    - this will build the zig shared library
    - NOTE: the shared library interface is always going to be slower than calling zig directly, see
      FUTUREWORK for target state.

PROS:
    - Hooks up to PET pipelines easily.
    - Good for small-medium datasets (<a few gigabytes unpacked).
    - Fast as long as chunksize and input data ctype conforms
    - Windows/mac supported if multiprocessing is disabled, but may depend on actual loaders
CONS:
    - Not good for largescale operational use.
    - Uses cffi and external backend, requiring additional compatiblity layer being maintained.
    - The zig backend (in this example) does not do parallel compute.

FUTUREWORK:
    - The preferrable pattern is to use a listener pattern using zeromq or similar.

    - The listener pattern slower interface than ffi for shared memory computations

    - HOWEVER, it will be MUCH faster if the data loading also happens in zig, since the only thing
      passed via zeromq is the worker task definition. This is a reasonable tradeoff.

    - The other advantage is that the compatiblity layers above do not need to be maintained for
      each backend,

    - instead a single compatiblity layer (e.g. a protobuf definition or similar) is all that's
      required for different backends to interact with one another.
"""

from persistence.methods._median import _median_of_three_zig


def _mock_dataset():
    """
    A mock dataset used for testing. This is a relatively small array so it may not benefit much
    from parallelism. The actual example should be running on real satellite data.

    x, y, time, ensembles, levels
    (intentional) suboptimal placement with ensembles/levels at the end.
    - 500, 500 is a reasonable size for a region
    - 24 => typical of hourly data
    - 3 ensembles
    - 3 levels
    """
    # these could also be in main guard, but just being explicit
    import numpy as np
    import xarray as xr

    # --- Uh Oh! ---
    # shape_input1 = (500, 500, 24, 99, 168)
    # ---
    # NOTE: setting the above would give you this impressive warning which is worth understanding:
    # ```
    #   numpy._core._exceptions._ArrayMemoryError: Unable to allocate 744. GiB for an array with
    #   shape (500, 500, 24, 99, 128) and data type float64
    # ```
    # The reason why is important is that you may _think_ this is a reasonably small dataset, and on
    # disk it may actually just be stored as 1GB or even less maybe 20MB the reason is:
    # 1. bit packing
    # 2. compression
    # 3. np.nan is not the same as "nothing", otherwise the structural integrity of the array will
    #    collapse. Sparse arrays will need a sparse array paradigm, but that will also make things
    #    complicated in the backend.
    # 4. wait but this is mocking it in memory, my dataset will be chunked!
    # ---
    shape_input1 = (500, 500, 9, 24, 6)
    shape_input2 = (400, 400, 9)
    shape_input3 = (5, 2, 9, 2)  # for manual inspection
    dimnames1 = ("x1", "y1", "time", "n_ens", "levels")
    dimnames2 = ("x2", "y2", "time")
    dimnames3 = ("k", "x3", "time", "y3")
    name_varA = "varA"
    name_varB = "varB"
    name_varC = "varC"

    # set unique rng context and constant seed for reprodicibility
    rng_context = np.random.default_rng(seed=42)
    arr1 = rng_context.random(list(shape_input1))
    arr2 = rng_context.random(list(shape_input2))
    arr3 = np.array(
        [
            [
                [[1.0, -100.0], [4.0, -20.0], [-1.0, -200.0]] * 3,
                [[1.0, 20.0], [1.0, 5.0], [1.0, 4.5]] * 3,
            ],
            [
                [[1.0, -100.0], [4.0, -20.0], [-1.0, -200.0]] * 3,
                [[3.0, 20.0], [1.0, 5.0], [5.0, 4.5]] * 3,
            ],
            [
                [[-4.0, -100.0], [4.0, -20.0], [-1.0, -200.0]] * 3,
                [[3.0, 2.0], [1.0, 5.0], [5.0, 4.5]] * 3,
            ],
            [
                [[1.0, -100.0], [1.0, -75.0], [1.0, -100.0]] * 3,
                [[-4.0, -147.0], [4.0, -20.0], [-1.0, -200.0]] * 3,
            ],
            [
                [[30.0, -100.0], [1.0, -75.0], [1.0, -100.0]] * 3,
                [[-4.0, -147.0], [-17.0, -20.0], [-1.0, -68.0]] * 3,
            ],
        ]
    )
    print(arr3.shape)

    # make dataset from numpy data above and dims, assume the dim names are common and taken from left
    # to right, i.e. either A in B and/or B in A without loss of generality.
    ds_mock = xr.Dataset(
        {
            name_varA: xr.DataArray(arr1, dims=dimnames1),
            name_varB: xr.DataArray(arr2, dims=dimnames2),
            name_varC: xr.DataArray(arr3, dims=dimnames3),
        }
    )

    return ds_mock


def run_example(ds_input, use_real=True, backend="zig", num_workers=1, num_chunks=1):
    # TODO: use library directly under main guard
    print("Example: python with zig.")
    print("---")
    print("NOTE: Optionally uses satellite data if appropriate nci group")

    # NOTE: scoped import so that context isn't leaked - being safe here though it is likely okay
    # for this to be on the global scope or at the very least main guarded is sufficient.
    from persistence import persistence_impl

    if use_real:
        NotImplementedError("mechanism to run real satellite data not yet implemented")
    else:
        # ---
        # some printing logic for display/debugging
        print("using mock data... use_real=False")
        print("\n--- mocking data ---")
        for v, da in ds_input.data_vars.items():
            print("...")
            for i, (n, s) in enumerate(zip(da.dims, da.shape)):
                print(f"{v}:shape={n}={s}")
        print("---")
        # ---

        # ---
        # TODO:
        # There is a flaw here if time index is not always the first index, since there is no
        # guarantee that the datasets share the array dimensions - this needs to be rectified.
        #
        # This can be done by requesting named index for time at the higher level api instead of the
        # integer directly. This is only really necessary for datasets and is infact insufficient
        # for numpy.
        #
        # We still need `idx_time_dim` for `numpy` support, so it'll have to be a mutually
        # exclusive argument.
        #
        # For testing purposes, this is a lower priority since the user can always just stick to
        # data arrays and computing each variable separately in a for loop wrapper with minimal loss
        # to performance, since the variable count is not likely to be very large.
        import time

        ts = time.time()
        print(f"ts={ts}")
        ds_output = persistence_impl.predict(
            ds_input,
            idx_time_dim=list(ds_input.dims).index("time"),
            num_workers=num_workers,
            # 20 chunks/2 workers => 10% of the data is loaded at any given time (assuming optimal chunking)
            num_chunks=num_chunks,
            method="median_of_three",
            simple_impute=False,
            backend_type=backend,
        )

        # ---
        te = time.time()
        print(f"te={te}")
        print(f"total={te - ts}s")
        print(f"size={ds_output.sizes}")
        print("---")
        print(ds_output)
        return ds_output


if __name__ == "__main__":
    import multiprocessing

    # ---
    # Notes:
    # - WHEN IN DOUBT: set num_chunks = 1 and num_workers = 1
    #
    # - IF USING DATASETS: chunk strategy is the SAME between variables. This could be very slow for
    #   certain variables that are very small in data size.
    #
    # - Support for datasets is for convenience only NOT SPEED. Supported settings != optimal settings
    #
    # - FASTER: use dataarrays or numpy arrays as inputs and combine later. This also allows the
    #   user to invoke embarassing parallelism at a higher level, and also choose different
    #   backend/computations for different variables.
    #
    # - (Not yet implemented) EVEN FASTER: data loading is also externally performed (FUTUREWORK).
    #   This allows for chunks to be stored on disk and retrieved by any compute engine, either
    #   using the same backend, a different backend, or using PET's existing computational stack
    #   (xarray + dask + numpy). The important take-away here is the separation of concern allows
    #   for flexiblity and portability.
    #
    # CAUTION: windows/mac - see WHEN IN DOUBT above, except it applies ALMOST ALWAYS.
    # ---
    NUM_WORKERS = 1
    NUM_CHUNKS = 5

    try:
        multiprocessing.set_start_method("forkserver")
        print("Start method set to 'forkserver'")
    except RuntimeError as e:
        print(f"Could not set start method: {e}")

    ds_input = _mock_dataset()
    ds_output1 = run_example(
        ds_input,
        use_real=False,
        backend="zig",
        num_workers=NUM_CHUNKS,
        num_chunks=NUM_CHUNKS,
    )
    # NOTE: second run can be a bit faster as it likely does some caching, so actual times (not
    # shown) can be much slower (depends). This part isn't for speed/memory benchmarking reasons
    # rather for comparing outputs are equal.
    ds_output2 = run_example(
        ds_input,
        use_real=False,
        backend="numpy",
        num_workers=NUM_WORKERS,
        num_chunks=NUM_CHUNKS,
    )

    import numpy as np

    # to check equivilence mostly for random
    print(np.allclose(ds_output1.varA, ds_output2.varA))
    print(np.allclose(ds_output1.varB, ds_output2.varB))
    print(np.allclose(ds_output1.varC, ds_output2.varC))

    # for manual inspection
    print(ds_output1.varC)
    print(ds_output2.varC)
