# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


"""
This class contains reprojection methods to convert latlon data to and from HEALPix data. In this implementation, the
HEALPix structure is translated from its 1D array into a 3D array structure [F, H, W], where F=12 is the number of
faces and H=W=nside of the HEALPix map. The HEALPix base faces are indiced as follows


         HEALPix                              Face order                 3D array representation
                                                                            -----------------
--------------------------               //\\  //\\  //\\  //\\             |   |   |   |   |
|| 0  |  1  |  2  |  3  ||              //  \\//  \\//  \\//  \\            |0  |1  |2  |3  |
|\\  //\\  //\\  //\\  //|             /\\0 //\\1 //\\2 //\\3 //            -----------------
| \\//  \\//  \\//  \\// |            // \\//  \\//  \\//  \\//             |   |   |   |   |
|4//\\5 //\\6 //\\7 //\\4|            \\4//\\5 //\\6 //\\7 //\\             |4  |5  |6  |7  |
|//  \\//  \\//  \\//  \\|             \\/  \\//  \\//  \\//  \\            -----------------
|| 8  |  9  |  10 |  11  |              \\8 //\\9 //\\10//\\11//            |   |   |   |   |
--------------------------               \\//  \\//  \\//  \\//             |8  |9  |10 |11 |
                                                                            -----------------
                                    "\\" are top and bottom, whereas
                                    "//" are left and right borders


Details on the HEALPix can be found at https://iopscience.iop.org/article/10.1086/427976
"""


from typing import TypeVar, Literal, Optional

import functools
import warnings

import numpy as np
import healpy as hp
import xarray as xr
import reproject as rp
import astropy as ap

import edit.data
from edit.pipeline.warnings import PipelineWarning

from .base import BaseRemap


XR_TYPE = TypeVar("XR_TYPE", xr.Dataset, xr.DataArray)

HEALPIX_COORDS = ["face", "height", "width"]


class HEALPix(BaseRemap):
    """
    HEALPix Remapper
    """

    def __init__(
        self,
        spatial_coords: dict[str, int],
        nside: int,
        order: str = "bilinear",
        resolution_factor: float = 1.0,
        include_coords: bool = False,
        manual_rechunking: bool = True,  # Useful for small datasets
        template_dataset: Optional[str] = None,
        check_for_nans: bool = False,
    ):
        """
        HEALPix mesh Remapper as `pipeline` operations

        Args:
            spatial_coords (dict[str, int]):
                Dictionary of spatial coords to remap over, with the associated size.
            nside (int):
                The number of pixels each HEALPix face sides has. Must be power of 2.
            order (str, optional):
                The interpolation scheme ("nearest-neighbor", "bilinear", "biquadratic", "bicubic"),. Defaults to "bilinear".
            resolution_factor (float, optional):
                In some cases, when choosing nside "too large" for the source data, the
                projection can contain NaN values. Choosing a resolution_factor > 1.0 can resolve this but requires careful
                inspection of the projected data. Defaults to 1.0.
            include_coords (bool, optional):
                Include spatial_coords as variables for each face. Defaults to False.
            manual_rechunking (bool, optional):
                Manually rechunk to one chunk per spatial grid. Defaults to True.
            check_for_nans (bool, optional):
                Check for nans after remapping. Defaults to False.

        Raises:
            ValueError:
                If `spatial_coords` is wrong

        Examples:
            Remap ERA5 resolution data to faces of size 128.
            ```python
            import edit.pipeline
            import edit.data

            remapper = edit.pipeline.operations.xarray.remapping.HEALPix({'latitude':721, 'longitude':1440}, nside = 128)
            remapper.remap(edit.data.archive.ERA5.sample()['2000-01-01T00'])
            # Remapped data

            edit.pipeline.Pipeline(
                edit.data.archive.ERA5.sample(),
                remapper
            )
            ```
        """
        super().__init__()
        self.record_initialisation()

        if len(spatial_coords.keys()) != 2:
            raise ValueError(
                f"`spatial_coords` must be a two element dictionary for each spatial dimensions, not {spatial_coords}."
            )

        self.spatial_coords = spatial_coords

        self.nside = nside
        self.order = order
        self.nested = True  # RING representation not supported in this implementation
        self.include_coords = include_coords
        self.manual_rechunking = manual_rechunking
        self.check_for_nans = check_for_nans

        self._template_cache = edit.data.patterns.ArgumentExpansion("temp")
        if template_dataset is not None:
            self._template_cache.save(xr.open_dataset(template_dataset), "template")

        resolution = 360.0 / list(spatial_coords.values())[1]
        self.npix = hp.nside2npix(nside)

        # Define and generate world coordinate systems (wcs) for forward and backward mapping. More information at
        # https://github.com/astropy/reproject/issues/87
        # https://docs.astropy.org/en/latest/wcs/supported_projections.html
        wcs_input_dict = {
            "CTYPE1": "RA---CAR",  # can be further specified with, e.g., RA---MOL, GLON-MOL, ELON-MOL
            "CUNIT1": "deg",
            "CDELT1": -resolution * resolution_factor,  # -r produces for some reason less NaNs
            "CRPIX1": (list(spatial_coords.values())[1]) / 2,
            "CRVAL1": 180.0,
            "NAXIS1": list(spatial_coords.values())[1],  # does not seem to have an effect
            "CTYPE2": "DEC--CAR",  # can be further specified with, e.g., DEC--MOL, GLAT-MOL, ELAT-MOL
            "CUNIT2": "deg",
            "CDELT2": -resolution,
            "CRPIX2": (list(spatial_coords.values())[0] + 1) / 2,
            "CRVAL2": 0.0,
            "NAXIS2": list(spatial_coords.values())[0],
        }
        self.wcs_ll2hpx = ap.wcs.WCS(wcs_input_dict)

        wcs_input_dict = {
            "CTYPE1": "RA---CAR",  # can be further specified with, e.g., RA---MOL, GLON-MOL, ELON-MOL
            "CUNIT1": "deg",
            "CDELT1": resolution * resolution_factor,
            "CRPIX1": (list(spatial_coords.values())[1]) / 2,
            "CRVAL1": 179.0,
            "NAXIS1": list(spatial_coords.values())[1],
            "CTYPE2": "DEC--CAR",  # can be further specified with, e.g., DEC--MOL, GLAT-MOL, ELAT-MOL
            "CUNIT2": "deg",
            "CDELT2": resolution,
            "CRPIX2": (list(spatial_coords.values())[0] + 1) / 2,
            "CRVAL2": 0.0,
            "NAXIS2": list(spatial_coords.values())[0],
        }
        self.wcs_hpx2ll = ap.wcs.WCS(wcs_input_dict)

    def remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Remap `sample` from Lat Lon Grid to HEALPix mesh
        """

        coords: dict[Literal["face", "height", "width"], np.ndarray] = {}

        coords["face"] = np.array(range(12), dtype=np.int64)
        coords["height"] = np.array(range(self.nside), dtype=np.int64)
        coords["width"] = np.array(range(self.nside), dtype=np.int64)

        spatial_coords = list(self.spatial_coords.keys())

        if not self._template_cache.exists("template"):
            if not isinstance(sample, xr.Dataset):
                sample.to_dataset()
            self._template_cache.save(sample[spatial_coords], "template")

        if self.manual_rechunking:
            sample = sample.chunk(
                {
                    **{c: len(sample[c]) for c in spatial_coords},
                    **{d: 1 for d in set(sample.dims).difference(spatial_coords)},
                }
            )

        healpix_sample: XR_TYPE = xr.apply_ufunc(
            self.ll2hpx,
            sample,
            input_core_dims=[spatial_coords],
            output_core_dims=[HEALPIX_COORDS],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={
                "allow_rechunk": not self.manual_rechunking,
                "output_sizes": {"face": 12, "height": self.nside, "width": self.nside},
            },
            output_dtypes=[np.float32],
        )
        healpix_sample = healpix_sample.assign_coords(coords)

        if self.include_coords:
            face_coords = tuple(map(functools.partial(self.hpx1d2hpx3d, dtype=np.float64), reversed(hp.pix2ang(self.nside, range(self.npix), nest=True, lonlat=True))))  # type: ignore

            for coord, data in zip(self.spatial_coords.keys(), face_coords):
                healpix_sample[coord] = (HEALPIX_COORDS, data)

        return healpix_sample

    def inverse_remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Remap `sample` from HEALPix mesh to Lat Lon Grid
        """
        spatial_coords = list(self.spatial_coords.keys())

        if self.manual_rechunking:
            sample = sample.chunk(
                {
                    **{c: len(sample[c]) for c in HEALPIX_COORDS},
                    **{d: 1 for d in set(sample.dims).difference(HEALPIX_COORDS)},
                }
            )

        spatial_sample: XR_TYPE = xr.apply_ufunc(
            self.hpx2ll,
            sample,
            input_core_dims=[HEALPIX_COORDS],
            output_core_dims=[spatial_coords],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": not self.manual_rechunking, "output_sizes": self.spatial_coords},
            output_dtypes=[np.float32],
        )

        if self._template_cache.exists("template"):
            template = self._template_cache("template")
            spatial_sample = spatial_sample.assign_coords({coord: template[coord] for coord in spatial_coords})
        else:
            warnings.warn("Could not find template to rebuild coords from. May cause issues", PipelineWarning)

        return spatial_sample

    ### ******* ALL BELOW COPIED FROM zephyr/data_processing/remap/healpix.py WITH MINIMAL MODIFICATION ******* ###
    # MIT License

    # Copyright (c) 2018 Jonathan Weyn

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    def ll2hpx(self, data: np.ndarray) -> np.ndarray:
        """
        Projects a given array from latitude longitude into the HEALPix representation.

        :param data: The data of shape [height, width] in latlon format
        :return: An array of shape [f=12, h=nside, w=nside] containing the HEALPix data
        """
        # Flip data horizontally to use 'CDELT1 = -r' in the wcs for the reprojection below
        data = np.flip(data, axis=1)

        # Reproject latlon to HEALPix
        hpx1d, hpx1d_mask = rp.reproject_to_healpix(
            input_data=(data, self.wcs_ll2hpx),
            coord_system_out="icrs",
            nside=self.nside,
            order=self.order,
            nested=self.nested,
        )

        # Convert the 1D HEALPix array into an array of shape [faces=12, nside, nside]
        hpx3d = self.hpx1d2hpx3d(hpx1d=hpx1d)

        if self.check_for_nans:
            assert hpx1d_mask.all(), (
                "Found NaN in the projected data. This can occur when the resolution of the original data is too "
                "small for the chosen HEALPix grid. Increasing the 'resolution_factor' of the HEALPixRemap instance "
                "might help."
            )

        return hpx3d

    def hpx2ll(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Projects a given three dimensional HEALPix array to latitude longitude representation.

        :param data: The data of shape [faces=12, height=nside, width=nside] in HEALPix format
        :return: An array of shape [height=latitude, width=longitude] containing the latlon data
        """
        # Recompensate array indices [0, 0] representing top left and not bottom right corner (required for fyx2hpxidx)
        # data = data[[9, 8, 11, 10, 6, 5, 4, 7, 1, 0, 3, 2]]
        data = data[[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]

        # Convert the 3D [face, nside, nside] array back into the 1D HEALPix array
        hpx1d = self.hpx3d2hpx1d(hpx3d=data)

        # Project 1D HEALPix to LatLon
        ll2d, ll2d_mask = rp.reproject_from_healpix(
            input_data=(hpx1d, "icrs"),
            output_projection=self.wcs_hpx2ll,
            shape_out=(list(self.spatial_coords.values())[0], list(self.spatial_coords.values())[1]),
            nested=self.nested,
        )
        # ll2d = np.flip(ll2d, axis=1)  # Compensate flip in reprojection function above

        if self.check_for_nans:
            assert ll2d_mask.all(), (
                "Found NaN in the projected data. This can occur when the resolution of the "
                "HEALPix data is smaller than that of the target latlon grid."
            )
        return ll2d

    def hpx1d2hpx3d(self, hpx1d: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts a one-dimensional HEALPix array [NPix] into a three-dimensional HEALPix array of shape [F, H, W].

        :param hpx1d: The one-dimensional array in HEALPix convention
        :param dtype: The data type (float precision) of the returned array
        :return: The three-dimensional array in [F, H, W] convention
        """
        # Convert the 1D HEALPix array into an array of shape [faces=12, nside, nside]
        hpx3d = np.zeros(shape=(12, self.nside, self.nside), dtype=dtype)
        for hpxidx in range(self.npix):
            f, y, x = self.hpxidx2fyx(hpxidx=hpxidx)
            hpx3d[f, x, y] = hpx1d[hpxidx]

        # Compensate array indices [0, 0] representing top left and not bottom right corner (caused by hpxidx2fyx)
        return np.flip(hpx3d, axis=(1, 2))

    def hpx3d2hpx1d(self, hpx3d: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts a three-dimensional HEALPix array of shape [F, H, W] into a one-dimensional HEALPix array [NPix].

        :param hpx3d: The three dimensional array in HEALPix convention [F, H, W]
        :param dtype: The data type (float precision) of the returned array
        :return: The one-dimensional array in [NPix] HEALPix convention
        """
        hpx1d = np.zeros(self.npix, dtype=dtype)
        for f in range(12):
            for y in range(self.nside):
                for x in range(self.nside):
                    hpxidx = self.fyx2hpxidx(f=f, y=y, x=x)
                    hpx1d[hpxidx] = hpx3d[f, y, x]
        return hpx1d

    def hpxidx2fyx(self, hpxidx: int, dtype: np.dtype = np.float32) -> (int, int, int):
        """
        Determines the face (f), column (x), and row (y) indices for a given HEALPix index under consideration of the base
        face index [0, 1, ..., 11] and the number of pixels each HEALPix face side has (nside).

        :param hpxidx: The HEALPix index
        :return: A tuple containing the face, y, and x indices of the given HEALPix index
        """
        f = hpxidx // (self.nside**2)
        assert 0 <= f <= 11, "Face index must be within [0, 1, ..., 11]"

        # Get bit representation of hpxidx and split it into even and odd bits
        hpxidx = format(hpxidx % (self.nside**2), "b").zfill(self.nside)
        bits_eve = hpxidx[::2]
        bits_odd = hpxidx[1::2]

        # Compute row and column indices of the HEALPix index in the according face
        y = int(bits_eve, 2)
        x = int(bits_odd, 2)

        return (f, y, x)

    def fyx2hpxidx(self, f: int, x: int, y: int) -> int:
        """
        Computes the HEALPix index from a given face (f), row (y), and column (x) under consideration of the number of
        pixels along a HEALPix face (nside).

        :param f: The face index
        :param y: The local row index within the given face
        :param x: The local column index within the given face
        :return: The HEALPix index
        """

        # Determine even and odd bits of the HEALPix index from the row (y, even) and column (x, odd)
        bits_eve = format(y, "b").zfill(self.nside // 2)
        bits_odd = format(x, "b").zfill(self.nside // 2)

        # Alternatingly join the two bit lists. E.g., ["1", "0"] and ["1", "0"] becomes ["1", "1", "0", "0"]
        bitstring = ""
        for bit_idx in range(len(bits_eve)):
            bitstring += bits_eve[bit_idx]
            bitstring += bits_odd[bit_idx]

        return int(bitstring, 2) + f * self.nside**2
