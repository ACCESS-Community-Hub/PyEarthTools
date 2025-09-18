# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rainfields3 Accessor
"""

import functools
import logging
import os
import warnings
import zipfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pyearthtools.data
import pyproj
import xarray as xr
from pyearthtools.data.archive import register_archive
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.time import Petdt as petdt

from site_archive_nci.utilities import check_project

#: TODO: unused in this module - assumed to be used in tutorials.
SINGLE_RADAR_VARIABLES = ["RAIN", "DBZH"]
COMPOSITE_VARIABLES = ["prcp_crate"]

#: Each element is a 2-D array
NumpyMeshGrid2D = tuple[npt.ArrayLike, npt.ArrayLike]

#: The status prefix for constructing a status code
PROJECT_ERROR_STATUS_STR_SHORT = "PES"


class ErrorRadarProj(Exception):
    """
    Derived exception class specific to Rainfields3 projections.

    See: ProjErrorStatus
    """


class WarnRadarProj(UserWarning):
    """
    Derived warning class specific to Rainfields3 projections.
    """


class ProjErrorStatus(Enum):
    """
    Error states for radar projection conversions
    """

    #: metadata for projection not available in dataset
    ATTR_MISSING = 101

    #: metadata has unsupported projection kind (see ProjKind)
    PROJ_UNSUPPORTED = 102

    #: attr to pyproj failed - likely due to incompatible proj structure
    ATTR_TO_PYPROJ_CONVERSION_FAILED = 103

    #: inverse proj failed - likely due to invalid parameters
    INVERSE_PROJ_FAILED = 104

    #: failed to do interpolate - likely due to invalid parameters
    INTERPOLATE_PROJ_FAILED = 105

    #: unexpected error - likely due to implementation rather than user input
    UNEXPECTED_RUNTIME_ERROR = 999

    @staticmethod
    def errcode(es: "ProjErrorStatus") -> str:
        """
        A short form representation of the projection error status-code - for logging

        Returns:
            "PES-#" - where # is the integer representation of the enum code
        """
        ret_errcode = PROJECT_ERROR_STATUS_STR_SHORT + "-" + str(es.value)
        return ret_errcode


class ProjKind(Enum):
    """
    Types of projection that can be used.

    UNSUPPORTED = catch all for anything that is not supported
    """

    #: unsupported projection
    UNSUPPORTED = -1

    #: default. Also the only one supported.
    ALBERS_CONICAL_EQUAL_AREA = 0


class RadarProj(SimpleNamespace):
    """
    Namespace containing constants and functions related to radar projection.

    There are two main "external" functions in this namespace:
        1. `xy_to_lonlat` initiates the conversion routine.
        2. `make_lonlat_meshgrid` is a helper to create a lonlat_meshgrid.

        The rest are internal implementations.

    See docs on `RadarProj.xy_to_lonlat` for main usage.

    Assumptions - general:
        * EXPERIMENTAL (i.e. this namespace is tested but not widely used yet).
        * x-y coordinates are in kilometers (km).
        * lat-lon coordinates are in degrees (°).
        * any phyiscal bounds checks are offloaded to underlying pyproj libraries,
          it is the caller's responsibility to manage these.
        * only ALBERS_CONICAL_EQUAL_AREA is tested so far.

    Assumptions - interpolation:
        * if interp_lonlat=True, interpolation fill NaNs in points where the projection
          of lonlat_meshgrid exceeds the converted x-y coordinate bounds. This
          inherited from scipy and is intentionally not handled here.

        * however, within a multiple of the max x-y grid spacing
          (MAX_XY_EXTRAPOLATE_GRIDSIZE_MULT) away from the x-y boundaries interp,
          extrapolation is allowed. This is to address:
              a) rounding issues and
              b) preserve the non-NaN grid values
                 e.g. if the same grid sizes are required for x-y and lat-lon.

          see: scipy documentation to determine which methods allow for extrapolation.

        * if explicit extrapolation is required, this should be done elsewhere.

        * only rectilinear (rectangular with even/uneven spacing) x-y datasets
          are supported for interpolation.

        * interp may not attempt to preserve NaN values for simplicity reasons.

    TODO:

        Once we have determined that this namespace is useful we should move it to a
        more generic projection module or directly into PyEarthTools. (Currently
        only useful for Rainfields3.)
    """

    #: Multiplier for km to meters.
    KM_TO_M_MULTIPLIER = 1000

    #: Threshold within which extrapolation is allowed (multiple of x-y grid spacing)
    MAX_XY_EXTRAPOLATE_GRIDSIZE_MULT = 2

    #: Required name in projection attr - used to map to ProjKind
    REQUIRED_PROJATTR_GRIDMAPPINGNAME = "grid_mapping_name"

    #: group common pyproj exceptions
    PYPROJ_EXCEPTION_TRAPS = (
        pyproj.exceptions.CRSError,
        pyproj.exceptions.ProjError,
        ValueError,
        KeyError,
    )

    # extern
    @staticmethod
    def xy_to_lonlat(
        ds: xr.Dataset,
        *,
        interp_lonlat: bool = False,
        interp_method: str = "linear",
        lonlat_meshgrid: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
        force_proj: pyproj.Proj | None = None,
        proj_cache: list[pyproj.Proj] | None = None,  # mutable cache
        warn_on_different_proj: bool = False,
    ) -> xr.Dataset:
        """
        Main routine used to run the conversion process from x-y to lon-lat.

        The minimal usage would be to specify `ds` which will infer the projection
        from the dataset, but not do any interpolation or warn on inconsistent
        projections. Furthermore, the resultant lat-lon coordinates will be a 2-D grid
        dependent on x-y and will not generally be evenly spaced.

        There may be some advantages in considering the `interp*` options as they are
        suited for evenly spaced lat-lon meshgrids. This will make decomposing lat/lon
        computations easier (depending on usecase). If `interp_lonlat=True`,
        `lonlat_meshgrid` and `interp_method` (default: linear) must be specified.

        `force_proj` is useful if the projection information is not present in the data
        collection, but is known to be consistent (caller's responsibility).

        providing `proj_cache` and setting `warn_on_different_proj` will fill the cache
        (mutate) and raise warnings if inconsistent projections are detected (it is
        caller's responsibility to initialize and manage this cache).

        See argument docs for more info on configurability.

        Args:
            force_proj:
                By default proj is inferred from `ds` (`force_proj=None`);
                if `force_proj` _is not_ None, it will force all projections to use
                that.  Setting `force_proj` supersedes the following settings, which
                will no longer have any effect:
                    * `proj_cache`
                    * `warn_on_different_proj`
                default: None

            interp_lonlat:
                Whether to use the provided lonlat_meshgrid to perform forward
                projection and interpolation instead of projection inversion using x-y;
                default: False

            lonlat_meshgrid:
                The lon-lat grid to interpolate to, must be a result from a call to
                `np.meshgrid` with appropriate lonlat extents - handled by caller.  Must
                NOT be `None` if `interp_lonlat=True`;
                default: None

            proj_cache:
                Cache containing projections during the lifetime of the caller.  This
                object is managed by the caller/application.  Ideally we should only
                have one projection per data collection.
                    * None       => no caching happens
                    * Some(list) => list should be initialized/provided by Caller
                                    and is mutated with observed projections.
                default: None

            warn_on_different_proj:
                Whether to warn if a new projection is detected that is different from
                what's in `proj_cache` (except when its empty).
                default: False

        Returns:
            `xr.Dataset` with lon and lat coordinates added. x-y are kept - since
            lat-lon are dependent variables and are not necessarily evenly spaced

        Raises:
            ErrorRadarProj: see defined enum for various status codes. These range from
                attributes in the dataset to errors in projection computations.

            Inputs are not explicitly sanity checked against physical constraints. But
            projection libraries that are used internally will likely raise exceptions
            against this.

            Warns against inconsistent projections (warn_on_different_proj=True), but
            does not prevent any transformations (for now).

        .. note::

            The caller has to decide to drop x-y if needed e.g. if `interp_lonlat=True`
            and the provided meshgrid is evenly spaced.

        .. tip::

            `RadarProj._map_xy_meshgrid_to_lonlat_grid` can be used to create a meshgrid
            from evenly spaced lat/lon extents. see method docs for details.

            This function does not mandate the use of this, as the user may
            want to specify uneven grids.

            Further, to match latlon coordinates from another dataset one can
            input the following:

            .. code-block ::

                RadarProj.xy_to_lonlat(
                    # ... args/kwargs
                    interp_lonlat=True,
                    lonlat_meshgrid=np.meshgrid(ds_other.lon, ds_other.lat),
                    # ... more kwargs
                )
        """
        # variables
        rp = RadarProj  # alias
        ds_ret = None  # output
        projattr = None  # dict representation of projection - extracted from `ds`
        pyprojobj = None  # projection object used for computations i.e. pyproj.Proj
        projkind = ProjKind.UNSUPPORTED  # kind of projection

        # argument errors - handled directly, not by _handle_error_state
        if interp_lonlat and lonlat_meshgrid is None:
            raise ValueError(
                "lonlat_meshgrid must be specified, if interp_latlon=True. See:"
                " numpy.meshgrid on how to create this 2-D meshgrid."
            )

            test_lonlatmesh = (
                isinstance(lonlat_meshgrid, tuple)
                and len(lonlat_meshgrid) == 2
                and isinstance(lonlat_meshgrid[0], np.ArrayLike)
                and isinstance(lonlat_meshgrid[1], np.ArrayLike)
            )

            if not test_lonlatmesh:
                raise ValueError("Invalid lonlat_meshgrid specified. See numpy.meshgrid on how to" " create this")

        # ------------------------
        #  1. extract proj object
        # ------------------------
        # default: infer from dataset (force_proj is None)
        # force_proj: ignores any projattr in dataset and uses force_proj;
        #             does not warn against or cache inconsistent projections.

        if force_proj is not None:
            if not isinstance(force_proj, pyproj.Proj):
                raise ValueError("`force_proj` MUST be a `pyproj.Proj` object or None")

            # assign pyprojobj
            pyprojobj = force_proj

            # test projection kind is valid/compatible
            projattr = pyprojobj.crs.to_cf()
            projkind = rp._get_projkind_from_projattr(projattr)
            if projkind == ProjKind.UNSUPPORTED:
                rp._handle_error_state(ProjErrorStatus.PROJ_UNSUPPORTED)
        else:
            # extract from dataset
            projattr = rp._extract_projattr_from_ds(ds)
            if projattr is None:
                rp._handle_error_state(ProjErrorStatus.ATTR_MISSING)

            # test projection kind is valid/compatible
            projkind = rp._get_projkind_from_projattr(projattr)
            if projkind == ProjKind.UNSUPPORTED:
                rp._handle_error_state(ProjErrorStatus.PROJ_UNSUPPORTED)

            # transform to pyproj.Proj object to handle calculations
            _res = rp._transform_projattr_to_pyprojobj(projattr)
            if _res is None:
                rp._handle_error_state(ProjErrorStatus.ATTR_TO_PYPROJ_CONVERSION_FAILED)

            # store result if no errors found
            pyprojobj, _unused_crsobj = _res

            # only warn and mutate cache if it is specified
            if proj_cache is not None:
                rp._warn_inconsistent_proj_mut(pyprojobj, proj_cache, do_warn=warn_on_different_proj)

        # safety - we should have a projection by now or have raised an error.
        assert pyprojobj is not None

        # ----------------------
        #  2. convert to latlon
        # ----------------------
        # default: interp_lonlat = False => 2-D lon-lat coord grid with raw data
        # otherwise: interp_latlon = True => 1-D lon-lat coords with interpolated data

        # a. interp_lonlat = False => invert xy projection to get latlon (default)
        if not interp_lonlat:
            _res = rp._map_xy_meshgrid_to_lonlat_grid(ds, pyprojobj)
            if _res is None:
                rp._handle_error_state(ProjErrorStatus.INVERSE_PROJ_FAILED)
            ds_ret, _unused_latlon_grid, _unused_xy_meshgrid = _res

        # b. interp_lonlat = True => project latlon and interpolate xy to match
        else:
            _res = rp._interp_xy_grid_from_lonlat_meshgrid(ds, pyprojobj, lonlat_meshgrid, interp_method)
            if _res is None:
                rp._handle_error_state(ProjErrorStatus.INTERPOLATE_PROJ_FAILED)
            ds_ret, _unused_latlon_grid, _unused_xy_grid = _res

        # safety check - should not actually reach this.
        if ds_ret is None:
            rp._handle_error_state(ProjErrorStatus.UNEXPECTED_RUNTIME_ERROR)

        return ds_ret

    # extern
    @staticmethod
    def make_lonlat_meshgrid(
        *,  # keyword only to avoid mistyping
        lon_extent: tuple[float, float],
        num_lon: int,
        lat_extent: tuple[float, float],
        num_lat: int,
        endpoint: bool = False,
    ) -> NumpyMeshGrid2D:
        """
        Creates a meshgrid of evenly spaced lat-lon points. spaced by roughly
        `(lon_extent / num_lon)` by `(lat_extent / num_lat)`.

        lat-lon coordinates are in degrees

        Args:
            lon_extent: pair containing [start, stop) for lon.
            num_lon: number of lat points
            lat_extent: ditto for lat
            num_lat: ditto for lat
            endpoint: False by default. If True, endpoint (stop) of each lat
                lon extent will be treated as closed intervals that is:
                    - False => [start, stop)  (default)
                    - True  => [start, stop]

        Uses `np.linspace` defaults to open interval [start, stop). If stop needs to be
        included caller should add a small offset to stop.

        Returns:
            Pair of arrays containing the lon & lat values for each index in the
            `num_lon` by `num_lat` meshgrid.
        """
        lon_1d = np.linspace(*lon_extent, num=num_lon, endpoint=endpoint)
        lat_1d = np.linspace(*lat_extent, num=num_lat, endpoint=endpoint)
        lonlat_meshgrid = np.meshgrid(lon_1d, lat_1d)

        return lonlat_meshgrid

    @staticmethod
    def _warn_inconsistent_proj_mut(
        pyprojobj: pyproj.Proj,
        proj_cache: list[pyproj.Proj],  # mutable cache
        *,
        do_warn=False,  # do_warn because `warn` can conflict
    ):
        """
        Tests if the cache contains the projection, if not it adds it to the cache
        provided by the user/caller.

        This is more of a safety precaution against inconsistent projections, usually we
        expect to handle one projection at a time for a collection of data inputs. But
        there are times where it may be appropriate to have multiple.

        It's up to the application to maintain the cache. So `proj_cache` is mutated
        rather than returned.

        No guarentees are provided for computations on inconsistent projections. A
        warning is raised if `do_warn=True`

        FUTUREWORK: If there are enough cases of these warnings that it becomes an
        issue, we can add a handler in the data-accessor to manage invalid projections
        (either by filtering or throwing exception).
        """
        # safety
        assert isinstance(proj_cache, list)
        # Re: len(x) == 0, PEP20 > PEP8: blame numpy for this mess, and also
        #     foolish consistency, hobgoblins etc.
        empty_cache = len(proj_cache) == 0
        proj_def = pyprojobj.definition_string()
        proj_matched = any(
            map(
                lambda _p: _p.definition_string() == proj_def,
                proj_cache,
            )
        )
        warnobj = WarnRadarProj(
            f"Inconsistent projection found: {proj_def}. Either the proj_cache is"
            " stale, or the underlying radar data has multiple projections. This may"
            " or may not affect things depending on your usecase."
        )
        # mutate: extend instead of return so that it propagates up the call
        #         chain without needing to handle returns.
        if not proj_matched:
            proj_cache.extend([pyprojobj])
            # we don't warn for a empty_cache, that would be pointless
            # - just like this comment.
            if do_warn and not empty_cache:
                warnings.warn(warnobj)

    @staticmethod
    def _handle_error_state(es: ProjErrorStatus):
        """
        Handles or raises errors with appropriate messages.

        Currently an error state will always raise an exception
        i.e. terminates execution

        Args:
            es: The projection error status to be handled.

        Raises:
            ErrorRadarProj with appropriate error code depending on
            `es` (error status)
        """
        # literal names of ProjKind as a list
        # i.e. [ "UNSUPPORTED", "ALBERS_CONICAL_EQUAL_AREA", ... ]
        proj_kinds = [*map(lambda _p: _p.name, ProjKind)]

        # emulating switch case with dict because its more legible than if/else
        map_projerror_to_str = dict(
            [
                (
                    ProjErrorStatus.ATTR_MISSING,
                    "Could not find `proj` variable in dataset.",
                ),
                (
                    ProjErrorStatus.PROJ_UNSUPPORTED,
                    f"Unsupported projection. Supported: {proj_kinds}.",
                ),
                (
                    ProjErrorStatus.ATTR_TO_PYPROJ_CONVERSION_FAILED,
                    "Failed to convert extracted projection from dataset to pyproj object.",
                ),
                (
                    ProjErrorStatus.INVERSE_PROJ_FAILED,
                    ("Failed to invert x-y coordinates to lat-lon. Check that the proj" " in dataset is invertible."),
                ),
                (
                    ProjErrorStatus.INTERPOLATE_PROJ_FAILED,
                    (
                        "Failed to interpolate x-y grid to match the lat-lon meshgrid. Try"
                        " other interp methods or check that the meshgrid extents are"
                        " correct."
                    ),
                ),
                (
                    ProjErrorStatus.UNEXPECTED_RUNTIME_ERROR,
                    "Unexpected error - could not compute result - please raise an issue.",
                ),
            ]
        )

        # construct identifiable error code - for dev debugging
        error_str = map_projerror_to_str.get(es, None)
        error_code = ProjErrorStatus.errcode(es)
        error_str += f" [err={error_code}]"

        raise ErrorRadarProj(error_str)

    @staticmethod
    def _extract_projattr_from_ds(ds: xr.Dataset) -> dict | None:
        """
        Extracts projection data from dataset

        Returns:
            dict - containing projection attributes if extraction succeeds
            None - if attribute does not exist
        """
        ret = None

        if "proj" in ds.variables.keys() and RadarProj.REQUIRED_PROJATTR_GRIDMAPPINGNAME in ds.proj.attrs:
            ret = ds.proj.attrs

        return ret

    @staticmethod
    def _get_projkind_from_projattr(projattr: dict) -> ProjKind:
        """
        Maps the projection string to an Enum so that it can be symbolically
        handled in the code, instead of dealing with strings.

        Returns:
            ProjKind enum pointing to the type of projection.
            ProjKind.UNSUPPORTED if there was an issue deriving `ProjKind` or
            if the projection itself is not supported.
        """
        ret = ProjKind.UNSUPPORTED
        projkind_str = projattr.get(RadarProj.REQUIRED_PROJATTR_GRIDMAPPINGNAME, None)

        # case sensitivity shouldn't affect the kind of projection.
        if projkind_str is not None and projkind_str.lower() == "albers_conical_equal_area":
            ret = ProjKind.ALBERS_CONICAL_EQUAL_AREA

        return ret

    @staticmethod
    def _transform_projattr_to_pyprojobj(
        projattr: dict,
    ) -> tuple[pyproj.Proj, pyproj.CRS] | None:
        """
        Transforms projattr dictionary to a interface that pyproj expects

        Data in array is stored in CF-compliant format. We can extract it back
        to CRS (Coordinate Reference System) used by Proj using its helper.

        See: https://pyproj4.github.io/pyproj/stable/build_crs_cf.html#importing-crs-from-cf

        Returns:
            Pair (tuple) containing the PyProj object and the CRS object. CRS object is
            mainly for debugging.
        """
        ret = None

        try:
            crsobj = pyproj.CRS.from_cf(projattr)
            pyprojobj = pyproj.Proj(crsobj)
            ret = (pyprojobj, crsobj)
        except RadarProj.PYPROJ_EXCEPTION_TRAPS as _e:
            logging.exception(_e)
            return None  # signal failure to main handler

        return ret

    def _make_xycoord_from_latlon_meshgrid(
        x_grid: npt.ArrayLike,
        y_grid: npt.ArrayLike,
        lon_grid: npt.ArrayLike,
        lat_grid: npt.ArrayLike,
    ) -> tuple[xr.DataArray, xr.DataArray] | None:
        """
        x-y coordinates are in km
        lat-lon coordinates are in degrees
        """
        ret = None

        # --- 1-d grid extraction ---
        #
        # if lonlat is a meshgrid:
        #     lon_grid => rows are the same
        #     lat_grid => columns are the same
        lon_1d = lon_grid[0, :]
        lat_1d = lat_grid[:, 0]
        # ---

        # to account for pesky floating point issues - not perfect but good enough
        # if anything, keeping this weak means interpolation can still trigger instead
        # of overzealously raising errors.
        approx_unique = lambda _v: np.unique(np.round(_v * 1e6) // 1e6)  # noqa

        # check that the conversion is indeed unique.
        if (approx_unique(lon_1d) == approx_unique(lon_grid)).all() or (
            approx_unique(lat_1d) == approx_unique(lat_grid)
        ).all():
            da_x = xr.DataArray(x_grid, dims=["lon", "lat"], coords={"lon": lon_1d, "lat": lat_1d})
            da_y = xr.DataArray(y_grid, dims=["lon", "lat"], coords={"lon": lon_1d, "lat": lat_1d})
            ret = (da_x, da_y)
        else:
            _e = "ERROR: provided meshgrid is malformed."
            logging.exception(ValueError(_e))
            return None

        return ret

    @staticmethod
    def _interp_xy_grid_from_lonlat_meshgrid(
        ds: xr.Dataset,
        pyprojobj: pyproj.Proj,
        lonlat_meshgrid: tuple[NumpyMeshGrid2D, NumpyMeshGrid2D],
        interp_method: str = "linear",
    ) -> tuple[xr.Dataset, NumpyMeshGrid2D, NumpyMeshGrid2D] | None:
        """
        Projects lon-lat meshgrid into x-y and interpolates the input dataset,
        essentially resampling the x-y dimensions to the new interpolated values
        matched to the lat-lon meshgrid.

        Requires scipy for multidimensional interpolation.

        .. note::

           `xr.Dataset.interp` auto inserts lat-lon.

        Args:
            ds: the dataset (in x-y)
            pyprojobj: the projection
            lonlat_meshgrid: the lat-lon meshgrid to project to x-y
            interp_method: the method used by `xarray.Dataset.interp`

        Returns:
            Triple (tuple) containing:
                * interpolated ds (first) - main result
                * lonlat grid (second) - for debugging
                * xy grid (third) - for debugging
        """

        ds_interp = None
        ret = None
        scipy_kwargs = None
        can_extrapolate = False  # don't extrapolate by default

        # need to transpose - because numpy and proj are inconsistent
        # proj  : (x, y) -> (lon[n_y, n_x], lat[n_y, n_x])
        # numpy : (x, y) -> (lon[n_x, n_y], lat[n_x, n_y])
        lon_grid, lat_grid = lonlat_meshgrid
        try:
            xyproj = pyprojobj(lon_grid.T, lat_grid.T, errcheck=True, radians=False)
        except pyproj.ProjError as _e:
            logging.exception(_e)
            return None

        # convert to km before interpolating
        x_proj, y_proj = map(lambda _v: _v / RadarProj.KM_TO_M_MULTIPLIER, xyproj)

        # resolve can_extrapolate? by checking gridspacing
        can_extrapolate = RadarProj._can_extrapolate(ds, x_grid=x_proj, y_grid=y_proj)

        if can_extrapolate:
            # NOTE: scipy.interpn needs this explicitly set to None for extrap
            scipy_kwargs = {"fill_value": None}

        # do projection
        da_xyproj = RadarProj._make_xycoord_from_latlon_meshgrid(x_proj, y_proj, lon_grid, lat_grid)

        if da_xyproj is None:
            return None

        # do interpolation (also adds lat-lon to ds implicitly)
        try:
            da_xproj, da_yproj = da_xyproj
            ds_interp = ds.interp(
                x=da_xproj,
                y=da_yproj,
                method=interp_method,
                kwargs=scipy_kwargs,
            )
            ret = tuple([ds_interp, (lon_grid, lat_grid), (x_proj, y_proj)])
        except ValueError as _e:  # scipy uses ValueErrror
            logging.exception(_e)
            return None

        return ret

    @staticmethod
    def _can_extrapolate(
        ds: xr.Dataset,
        *,
        x_grid: npt.ArrayLike,
        y_grid: npt.ArrayLike,
    ) -> bool:
        """
        Checks if the boundaries of x_grid and y_grid are within
        MAX_XY_EXTRAPOLATE_GRIDSIZE_MULT * grid_spacing, where grid_spacing is the max
        (potential) grid area in ds.

        Particularly relevant for _interp_xy_grid_from_lonlat_meshgrid.

        Args:
            ds: the source dataset
            x_grid: the x coordinate grid to check against
            y_grid: the y coordinate grid to check against

        Returns:
            (bool) whether or not extrapolation should be allowed when doing x-y ->
                   lat-lon conversions.

        """
        # define helpers to avoid repetition errors
        fn_maxgridsize = lambda _v: np.abs(np.max(_v[0:-1] - _v[1:]))  # noqa
        fn_minmaxdiff = lambda _v1, _v2: (  # noqa
            np.abs(np.min(_v1) - np.min(_v2)),
            np.abs(np.max(_v1) - np.max(_v2)),
        )

        # compute max allowable grid difference as a ratio of x-y grid area
        xy_maxgridsize = fn_maxgridsize(ds.x.values), fn_maxgridsize(ds.y.values)
        maxgridsize = np.prod(xy_maxgridsize) * RadarProj.MAX_XY_EXTRAPOLATE_GRIDSIZE_MULT

        # compute grid difference at boundaries
        diffgrid = (
            fn_minmaxdiff(x_grid, ds.x.values),
            fn_minmaxdiff(y_grid, ds.y.values),
        )
        diffgridsize = np.prod(diffgrid)

        # check max spacing at boundaries is below maxgridsize
        can_extrapolate = diffgridsize <= maxgridsize

        return can_extrapolate

    @staticmethod
    def _map_xy_meshgrid_to_lonlat_grid(
        ds: xr.Dataset,
        pyprojobj: pyproj.Proj,
    ) -> tuple[NumpyMeshGrid2D, NumpyMeshGrid2D] | None:
        """
        Maps xy grid from provided x-y coordinates to its lonlat inverse projection

        Args:
            ds: dataset with the xy coordinates to create the meshgrid from
                (assumed to be in kilometers)
            pyprojobj: the projection to invert xy coordinates to lonlat

        Returns:
            Triple (tuple) containing:
                * ds with lonlat from proj inverse (first)  - main result
                * lonlat grid (second)  - for debugging
                * xy grid (third) - for debugging

        .. note::

            * Both grids must have the same size (N by M)
            * Note that xy is a meshgrid because its coordinates are spaced equally. For
              lonlat this is not the case, because of the equal area constraint.
            * This means each x does not map to a unique Lon, the only thing guarenteed
              is that each xy maps to a unique lonlat. This means that each of lat, lon
              is a function of (x, y)
        """
        # convert units kilometers to meters for xy_meshgrid
        x_metres = ds.x * RadarProj.KM_TO_M_MULTIPLIER
        y_metres = ds.y * RadarProj.KM_TO_M_MULTIPLIER
        x_grid, y_grid = np.meshgrid(x_metres, y_metres)
        lon_grid, lat_grid = (None, None)
        ret = None

        if not pyprojobj.has_inverse:
            _e = f"ERROR: projection: {pyprojobj} does not have an inverse"
            logging.exception(ValueError(_e))
            return None  # signal failure to main handler

        try:
            # NOTE: inverse takes in x-y rather than lat-lon even though docs
            # have the args explicitly set to lat-lon
            lon_grid, lat_grid = pyprojobj(x_grid, y_grid, inverse=True, errcheck=True, radians=False)
        except RadarProj.PYPROJ_EXCEPTION_TRAPS as _e:
            logging.exception(_e)
            return None

        # safety
        assert not (lon_grid is None or lat_grid is None)

        # assign lon/lat coordinates
        # NOTE: lon/lat grids map rows to y => y,x
        ds_inv = ds.assign_coords(lon=(["y", "x"], lon_grid), lat=(["y", "x"], lat_grid))

        # construct return tuple
        ret = tuple([ds_inv, (lon_grid, lat_grid), (x_grid, y_grid)])

        return ret


@register_archive("Rainfields3")
class Rainfields3(ArchiveIndex):
    """ArchiveIndex for Rainfields3 Australia-wide radar mosiac 2km^2 (Ausm310)"""

    _PROJ_FN_NAME = "_proj_laton"

    @property
    def _desc_(self):
        return {
            "singleline": "Rainfields3 Australia-wide radar mosiac 2km^2 (Ausm310)",
            "Documentation": "https://dx.doi.org/10.25914/DTTK-H476",
        }

    @classmethod
    def init_with_lonlatproj(
        cls,
        variables,
        *,
        proj_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Alternate initialiser with latlon coord transformer added in.

        .. note::

            Due to the complex inheritence structure of archives, it is uncertain if
            changing the API of `__init__` will cause unintended side-effects.

            As such, lonlat projection (which is not an established transform yet) will
            be initialised via this alternative method, to minimise sideffects.

            This is likely not the most ideal way but good enough for now.

        Args:
            proj_kwargs: dictionary containing additional keyword arguments to pass to
                latlon projection.

        see:
            * RadarProj.xy_to_lonlat for documentation on kwargs to use for proj_kwargs
            * Rainfields3.__init__ for common args for Rainfields3
        """
        if proj_kwargs is None:
            proj_kwargs = {}
        # Create an empty container before any initialisation
        rd = object.__new__(cls, variables, **kwargs)
        # Assign attribute to point to projection function
        setattr(
            rd,
            cls._PROJ_FN_NAME,
            functools.partial(RadarProj.xy_to_lonlat, **proj_kwargs),
        )
        # Perform the normal initialisation flow and return object
        rd.__init__(variables, **kwargs)
        return rd

    @property
    def fn_lonlatproj(self) -> Callable | None:
        """
        Determines if the object has a projection to lonlat builtin.

        Returns:
            - Callable => the underlying latlon projection function
            - None => it doesn't therefore projection cannot be done.

        see: init_with_lonlatproj
        """
        return getattr(self, Rainfields3._PROJ_FN_NAME, None)

    def __init__(
        self,
        variables: list[str] | str,
        **kwargs,
    ):
        """
        Setup Radar Demo Accessor

        Args:
            variables: Variables to retrieve
        """
        check_project(project_code="rq0")
        self.variables = [variables] if isinstance(variables, str) else variables

        base_transform = pyearthtools.data.transforms.variables.Trim(variables)

        self.walk_cache = {}  # Caches filesystem walks for efficiency

        super().__init__(transforms=base_transform)

        self.record_initialisation()

    def load(self, *args, **kwargs) -> Any:
        """

        Parameters:
            args: The list of matched filenames that should be loaded
            kwargs:

        Returns:
            The xarray object containing the data
        """

        zp = args[0][0][0][0]  # FIXME bad nesting

        # TODO: this needs a context manager so that filehandlers do not leak
        # i.e. it should be closed if the mapped `ds` is no longer being used.
        file_like = zp.open(mode="rb")

        # TODO: investigate if this can be done with `open` rather than `load`
        # alternatively cache to disk first, to avoid unnecessary in-memory loads
        ds = xr.load_dataset(file_like, engine="h5netcdf")

        # perform projection if object was initialised with projection method.
        maybe_proj = self.fn_lonlatproj
        if maybe_proj is not None:
            ds = maybe_proj(ds)

        return ds

    def quick_walk(self, caldate):
        """
        Walking a large filesystem to find matching filenames can take a long time.
        This function uses the query dictionary to more effectively walk only
        the parts of the filesystem actually relevant to the dataset. If performance
        is not a concern or if the filesystem is small, just use os.walk.
        """

        RADAR_HOME = self.ROOT_DIRECTORIES["Rainfields3"]
        basepath = Path(RADAR_HOME)

        # Only walk the filesystem once, use the cache thereafter
        if caldate in self.walk_cache:
            cache = self.walk_cache[caldate]
            return cache

        # Only walk the filesystem for configured institutions
        walk_from = os.path.join(basepath, caldate)
        all_entries = list(os.walk(walk_from))
        cache = []

        for root, dirs, files in all_entries:
            # We don't care about directories with no files
            if files:
                cache.append((root, dirs, files))

        self.walk_cache[caldate] = cache
        return cache

    def filesystem(self, query_dictionary=None):
        """
        Given the supplied query, return all filenames which contain the data necessary
        to extract the data for the query. For example, figure out what time index and
        variables the user wants, and go find all the files matching those time indexes
        and variables so they can get loaded into memory and then the relevant data
        extracted and re-aggregated as needed by other parts of the system.

        An additional complication for the radar archives is that all the files for a
        particular day are zipped up together. This means that instead of filenames,
        this function will instead return tuples of zipfile objects and the
        path+filename to load from within that zipfile object.

        The load function will then retrieve the specified file into memory from the
        zipfile rather than with a simple file open operation.
        """
        paths = {}
        tuples_to_open = []
        if query_dictionary is None:
            query_dictionary = {}
        pdt = query_dictionary

        calpath = f"{pdt.year}"  # Rainfields3 data is organised into folders by year

        # Walk the filesystem finding relevant file paths
        # A more efficient walk ordered by time or primary dimension may be more efficient
        walked = self.quick_walk(calpath)
        for root, _dirs, files in walked:
            paths = [os.path.join(root, file) for file in files]
            relevant = [self.match_path(p, query_dictionary) for p in paths]
            relevant = [r for r in relevant if r]
            tuples_to_open += relevant

        return tuples_to_open

    def match_path(self, path, query):
        """
        Given a query (typically a date/time) and a full path to a filename, go and
        check if that filename is likely to contain data relevant to the query.

        .. note::

            This will only work for requests specified to the minute level.

        Returns:
            True if the path and query match
            False if the file should be ignored
        """

        _directory, filename = os.path.split(path)
        id_and_date, varname, _extension = filename.split(".")
        _radar_id, datepart = id_and_date.split("_")
        pdt = petdt(datepart)

        match_variables = [v for v in self.variables if v in varname]
        if not match_variables:
            return []

        # The zip files are grouped to the day only, then the files within to the minute
        # at resolution isn't working putting in a workaround for now # FIXME
        if pdt.year != query.year:
            return []

        if pdt.month != query.month:
            return []

        if pdt.day != query.day:
            return []

        # We are now into a matched zipfile for the query
        zp = zipfile.Path(path)

        contents = list(zp.iterdir())

        for zip_path in contents:
            nc_filename = zip_path.name
            if nc_filename == "_file_list":
                continue

            id_and_date, varname, extension = nc_filename.split(".")
            if extension != "nc":
                continue

            _radar_id, caldate, timepart = id_and_date.split("_")
            nc_datetime = petdt("T".join([caldate, timepart]))

            if nc_datetime.hour != query.hour:
                continue

            if nc_datetime.minute != query.minute:
                continue

            # By this stage, should have match year, month, day, hour and minute
            # Only one match should be possible, return here
            return [(zip_path, nc_filename)]

        # This should be unreachable, but could occur if the zipfile is empty
        raise IOError("Zipfile may be empty")
