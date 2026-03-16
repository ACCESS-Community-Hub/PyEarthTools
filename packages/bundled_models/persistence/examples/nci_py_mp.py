"""
This example is a WORK IN PROGRESS.

Use multiprocessing to delegate chunks to various processes, in order to compute the persistence
method in an embarassingly parallel fashion.

PROS:
    - Hooks up to PET pipelines easily.
    - Good for small-medium datasets (<a few gigabytes unpacked).
CONS:
    - Not good for largescale operational use.
    - Taylored for NCI use only.
    - Windows/mac support not guarenteed.
    - Data MUST be stored in a chunked fashion for multiprocessing to have any benefit, otherwise
      everything will be loaded in memory anyway, and its better to let numpy handle everything (set
      workers=1 and num_chunks=1). TODO: there likely should be a smart way to detect this.
    - Slow.
"""


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
    shape_input1 = (500, 500, 24, 10, 24)
    dimnames1 = ("x1", "y1", "time", "n_ens", "levels")
    dimnames2 = ("x2", "y2", "time")

    # without loss of generality specify two variables, they can have varying dimensions.
    # for arguments sake, change the shape of the second.
    shape_input2 = (400, 400, 24)
    name_varA = "varA"
    name_varB = "varB"

    # set unique rng context and constant seed for reprodicibility
    rng_context = np.random.default_rng(seed=42)
    arr1 = rng_context.random(list(shape_input1))
    arr2 = rng_context.random(list(shape_input2))

    # make dataset from numpy data above and dims, assume the dim names are common and taken from left
    # to right, i.e. either A in B and/or B in A without loss of generality.
    ds_mock = xr.Dataset(
        {
            name_varA: xr.DataArray(arr1, dims=dimnames1),
            name_varB: xr.DataArray(arr2, dims=dimnames2),
        }
    )

    return ds_mock


def run_example(ds_input, use_real=True, num_workers=1, num_chunks=1):
    # TODO: use library directly under main guard
    print("Example: python multiprocessing on nci.")
    print("---")
    print("NOTE: this example requires appropriate project data accessible")
    print("      it currently uses the satellite data (TODO: which nci group?)")

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
            backend_type="numpy",
        )

        # ---
        te = time.time()
        print(f"te={te}")
        print(f"total={te - ts}s")
        print(f"size={ds_output.sizes}")
        print("---")
        print(ds_output)


if __name__ == "__main__":
    import multiprocessing

    # CAUTION: windows/mac - this may not work, use num_workers=1 instead
    try:
        multiprocessing.set_start_method("forkserver")
        print("Start method set to 'forkserver'")
    except RuntimeError as e:
        print(f"Could not set start method: {e}")

    ds_input = _mock_dataset()
    run_example(ds_input, use_real=False, num_workers=1, num_chunks=1)
