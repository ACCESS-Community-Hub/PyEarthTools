# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import numpy as np

import pytest

from edit.utils.data import Tesselator

tesselator_tests = [
    # Standard Easy
    ((100, 100), (50, 50), None),
    ((100, 100), (50, 50), (25, 25)),
    # In divisible
    ((100, 100), (27, 27), None),
    ((100, 100), (27, 27), (13, 13)),
    ((100, 100), (50, 50), (13, 13)),
    # Larger than array
    ((100, 100), (51, 51), None),
    ((100, 100), (51, 51), (13, 13)),
    # Not Square
    ((100, 100), (16, 25), None),
    ((100, 100), (20, 25), None),
    # ((100, 100), (20, 25), (4, 5)),
    # ((100, 100), (20, 25), (5, 5)),
    # ((100, 100), (16, 25), (5, 5)),
    # ((100, 100), (16, 25), (5, 5)),
    ## TODO This will fail :(
    # Larger Arrays
    ((10, 100, 100), (50, 50), None),
    ((10, 100, 100), (25, 25), None),
    ((10, 100, 100), (27, 27), None),
    ((10, 100, 100), (50, 50), (25, 25)),
    ((10, 100, 100), (25, 25), (13, 13)),
]


@pytest.mark.parametrize("imgsize,kernel_size, stride_size", tesselator_tests)
def test_tesselator_horizontal(imgsize, kernel_size, stride_size):
    sub_image_size = [*imgsize[0:-2], imgsize[-2], imgsize[-1] // 2]
    fake_data = np.concatenate([np.zeros(sub_image_size), np.ones(sub_image_size)], axis=-1)
    assert imgsize == fake_data.shape

    tesselator = Tesselator(kernel_size=kernel_size, stride=stride_size)

    t_patches = tesselator.patch(fake_data)
    t_rebuild = tesselator.stitch(t_patches)

    assert fake_data.shape == t_rebuild.shape  # Shapes are the same
    assert (fake_data == t_rebuild).all()  # Values agree


@pytest.mark.parametrize("imgsize,kernel_size, stride_size", tesselator_tests)
def test_tesselator_vertical(imgsize, kernel_size, stride_size):
    sub_image_size = [*imgsize[0:-2], imgsize[-2] // 2, imgsize[-1]]
    fake_data = np.concatenate([np.zeros(sub_image_size), np.ones(sub_image_size)], axis=-2)
    assert imgsize == fake_data.shape

    tesselator = Tesselator(kernel_size=kernel_size, stride=stride_size)

    t_patches = tesselator.patch(fake_data)
    t_rebuild = tesselator.stitch(t_patches)

    assert fake_data.shape == t_rebuild.shape  # Shapes are the same
    assert (fake_data == t_rebuild).all()  # Values agree


# @pytest.mark.parametrize(
#     "imgsize,kernel_size, stride_size",
#     tesselator_tests
# )
# def test_tesselator_random(imgsize, kernel_size, stride_size):
#     rng = np.random.default_rng(seed=42)
#     fake_data = rng.random(imgsize)
#     assert imgsize == fake_data.shape

#     tesselator = Tesselator(kernel_size=kernel_size, stride=stride_size)

#     t_patches = tesselator.patch(fake_data)
#     t_rebuild = tesselator.stitch(t_patches)

#     assert fake_data.shape == t_rebuild.shape # Shapes are the same
#     assert ((fake_data - t_rebuild) == 0).all() # Values agree


def array_func(*args):
    coords = args[-2:]
    return coords[0] // 10 + coords[1] // 10


@pytest.mark.parametrize("imgsize,kernel_size, stride_size", tesselator_tests)
def test_tesselator_grid(imgsize, kernel_size, stride_size):
    fake_data = np.fromfunction(array_func, imgsize)

    assert imgsize == fake_data.shape

    tesselator = Tesselator(kernel_size=kernel_size, stride=stride_size)

    t_patches = tesselator.patch(fake_data)
    t_rebuild = tesselator.stitch(t_patches)

    assert fake_data.shape == t_rebuild.shape  # Shapes are the same
    assert (fake_data == t_rebuild).all()  # Values agree
