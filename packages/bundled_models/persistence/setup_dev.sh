#!/usr/bin/env bash

# setup build in include folder
(
cd src/persistence/include
rm -r lib/*
rm -r lib/*.a
rm -r __pycache__/
rm *.c
rm *.so
rm *.o
rm *.a
)

zig build --prefix src/persistence/include

# run cffi
(
cd src/persistence/include
# move shared libraries to same directory, required for runs
cp ./lib/* .
python _cffi.py
)
