from enum import StrEnum, auto


class PersistenceBackendType(StrEnum):
    """
    Enumeration of supported compute backends for persistence computations.

    ---

    SUPPORTED BACKENDS (as of 2026-02-28):
        - NUMPY (20260228)
        - others are WIP

    Note: "supported" implies that the backend is supported by the build system, it does not imply
           that the particular persistence method itself is supported for that backend.

    ---

    Backends are configured at the "build" level in pyproject.toml, e.g. for rust this may be
    maturin/pyO3, which usually handles most of the heavy lifting.

    numba might require certain system dependencies - e.g. llvm, to function since it requires
    building on the fly.

    For C/zig this would involve using:
        a. ziglang/zig-pypi to build the zig packages into wheels and running them on the fly using
           sys.execute to execute the wheel as a module, building/running zig on-the-fly. Avoids
           having to distribute the pre-built dependencies, but may not work well with specific
           interfaces like `numpy`.
        b. using setuptools-zig to build them into a "integrated" library and packaging the build
           into the wheel/distribution
        c. using cffi or ctypes.

    Methods a. and b. would require extending Python.h directly, and hence are preferrable, since
    they don't involve foreign calls. Unlike numba, method a. exists for zig where jit compilation
    can happen without dependency on additional system libraries.

    All of the above methods generally avoid (or at least have the ability to avoid) the need for
    conda environments and are pretty light weight.
    """

    C = "c"
    NUMBA = "numba"
    NUMPY = "numpy"
    RUST = "rust"
    ZIG = "zig"
    UNKNOWN = auto()

    def check_support(self):
        """
        As per the module documentation, this method only tells you if a particular backend is
        supported by the *build system*, it doesn't imply that the backend is useable for any given
        method.

        Therefore, this check can and should be done as early as possible. Whereas method
        compatiblilty will be checked later into the runtime but still early enough point in the
        code, before attempting the computation. (see `PersistenceCompute` for more details)
        """
        match self:
            case PersistenceBackendType.NUMPY:
                return
            case PersistenceBackendType.ZIG:
                return
            case _:
                raise NotImplementedError(
                    f"PersistenceBackendType: {self} is not supported"
                )
