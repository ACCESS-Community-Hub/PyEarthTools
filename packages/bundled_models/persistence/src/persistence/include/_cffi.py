"""
Compile cffi code and put them in the include directory
"""
from cffi import FFI
import sys
import os


_zig_c_declarations = """
float median_of_three(float, float, float);
"""
_zig_c_libname="libpersistence_zig"
_include_libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")

def compile_zig():
    # cffi
    ffibuilder = FFI()
    # this is for python to know about
    ffibuilder.cdef(_zig_c_declarations)
    # NOTE: this is needed for API mode (recommended)
    # no header here so declaration is repeated.
    ffibuilder.set_source(
        "_persistence_zig",
        _zig_c_declarations,
        libraries=["persistence_zig"],
        library_dirs=[_include_libdir],
        extra_link_args=[f"-Wl,-rpath={_include_libdir}"]
    )
    ffibuilder.compile(verbose=True)


if __name__ == "__main__":
    compile_zig()
