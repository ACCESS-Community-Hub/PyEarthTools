# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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
PET Projection x-y to lon-lat. Rectilinear lon-lat grids. Representable as 1-D
coordinates for ease of processing.

At the moment this module focuses on the Australian region, and on satellite and radar data
in the region. Over time, additional use cases will help to make this more generalisable.
"""

# ----------------------------------------------------------------------------
# UPDATES: nikeethr - 2025-07-25 21:27
# - basic workflow is ready to test
# - satellite was especially tricky to get right because of its projection
#   system
# - trying it out on the notebook is the next step
# - T2 and T5 are important to do as well (below)
# ----------------------------------------------------------------------------
# TODO: nikeethr - 2025-07-25 21:27
# - [T1]: test notebook
# - [T2]: see if we can download pre-canned transform grids that will work for our region
# - [T3]: proper unit tests
# - [T4]: speed profiling
# - [T5]: safety check on transform errors + revert to simpler transform
# ----------------------------------------------------------------------------

# --- built-in imports ---

import functools
import warnings

from dataclasses import dataclass, KW_ONLY
from enum import IntEnum, StrEnum, auto
from typing import NamedTuple

# --- package imports ---

import numpy as np
import numpy.typing as npt
import xarray as xr

import pyproj
from pyproj.enums import TransformDirection
from pyproj.transformer import Transformer
from pyproj.aoi import AreaOfInterest


# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------
#: Each element is a 2-D array
NumpyMeshGrid2D = tuple[npt.ArrayLike, npt.ArrayLike]

#: Multiplier for km to meters.
KM_TO_M_MULTIPLIER = 1000  # meters

#: Threshold within which extrapolation is allowed (multiple of x-y grid spacing)
MAX_XY_EXTRAPOLATE_GRIDSIZE_MULT = 2000  # meters

#: Tuple containing the reference projections from x[0] to x[1]
CRSPair = (pyproj.CRS, pyproj.CRS)

#: (CRSPair, Transformer)
CRSTransformPair = (CRSPair, Transformer)


# -----------------------------------------------------------------------------
# Errors/warning
# -----------------------------------------------------------------------------
#: group common pyproj exceptions
PYPROJ_EXCEPTION_TRAPS = (
    pyproj.exceptions.CRSError,
    pyproj.exceptions.ProjError,
    ValueError,
    KeyError,
)


class ProjKind(StrEnum):
    # standard domain for global lon-lat coordinates
    WGS84 = "ESPG:4326"
    # 3m accuracy in AUS region
    GDA94 = "EPSG:3577"
    # FUTURE-WORK: move these to a config - note: its better NOT to use these defaults if possible
    AEA_AUS = "proj=aea +lat_1=-36 +lat_2=-18 +lon_0=132 +units=m aea_aus"
    GEO141_AUS = "proj=geos +lon_0=140.7 +h=35785863 +x_0=0 y_0=0 +a=6378137 +rf=298.257024882273 +units=m no_defs"
    UNKNOWN = auto()


class ProjSource(IntEnum):
    """
    For simplicity all projection definitions are expected to abide by these
    standard units:
    - x-y is in metres
    - lon-lat is in degrees

    +------------------------------------------------------------------------
    | 1. AEA_RAINFIELDS3/AEA_AUS
    +------------------------------------------------------------------------
    | > projection scheme assumes AEA.
    | > this is also used as the target CRS for PET.
    | --- psuedo-schema
    | Radar:
    |     - id=310, merged mosiac of several radars
    |     - 2km resolution, AUS
    | Proj4:
    |     attribute: ds.proj.attrs
    |     default:
    +------------------------------------------------------------------------

    +------------------------------------------------------------------------
    | 2. HIMA8_HIMA9/GEOS141
    +------------------------------------------------------------------------
    | > projection scheme assumes GEOS141. Can optionally attempt to derive
    | > projection string from ds.geostationary.proj4.
    | > NOTE: some derived products may be exceptions to this.
    | --- psuedo-schema
    | Satellites:
    |     - Himawari08
    |     - Himawari09
    | ProjReference:
    |     - WGS84
    |     - GEOS141
    |     - https://proj.org/en/stable/operations/projections/geos.html
    | Proj4:
    |     attribute: ds.geostationary.proj4
    |     default: +proj=geos +lon_0=140.7 +h=35785863 +x_0=0 +y_0=0
    |              +a=6378137 +b=6356752.3 +units=m +no_defs
    | DefaultUnits:
    |     - meters (x,y,h)
    |     - degrees (lon,lat)
    | ObsProducts:
    |     Band3_ScaledRadiance_0_64_um:
    |         - 0.5km spatial resolution (only available early morning/late
    |           afternoon and night)
    |     Band7_BrightnessTemp_3_83_um:
    |         - 2km spatial resolution
    |     Band10_BrightnessTemp_7_40_um:
    |         - 2km spatial resolution
    |     Band13_BrightnessTemp_10_35_um:
    |         - 2km spatial resolution
    |     SolarZenithAngle:
    |         - 0.5km, 1km & 2km resolutions available
    | DerivedProducts:
    |     CloudMask:
    |         - day and night
    |         - uses nx/ny instead of x-y
    |     CloudOpticalThickness
    |         - not available in low light/solar zenith angles>60
    |         - uses nx/ny instead of x-y
    |     InstataneousSolarRadiation
    |         - no projection needed, in latlon
    +------------------------------------------------------------------------
    """

    # EPSG compliant generic definitions 1024 -> 32767
    # ---
    GDA94 = 3577  # Australias
    WGS84 = 4326  # World

    # Custom codes
    # ---

    # ------------------------------------------------------------------------
    # BoM Products codespace 1: 110000 -> 120000
    # CAUTION: These codes are not fixed yet - always use the enum directly
    # rather than the integer representation
    # --- radar ---
    AEA_RAINFIELDS3 = 101001
    # --- satellite ---
    HIMA8_HIMA9 = 102001
    # ---

    # unknow type
    UNKNOWN = auto()

    def proj_attr_exists(self, ds: xr.Dataset) -> bool:
        assert ds is not None
        if self == ProjSource.AEA_RAINFIELDS3:
            # check grid mapping attr (Radar/Rainfields3)
            return "proj" in ds and "grid_mapping_name" in ds["proj"].attrs
        if self == ProjSource.HIMA8_HIMA9:
            # check proj4 attr (Himawari/Satellite Products)
            return "geostationary" in ds and "proj4" in ds["geostationary"].attrs
        # only himawari and rainfields3 products currently can have derived
        # projection attributes
        return False

    def get_default_proj(self) -> pyproj.CRS:
        # FUTURE: use switch-case statements when support for python 3.10 is
        # removed
        # wgs84
        if self == ProjSource.WGS84:
            return pyproj.CRS(str(ProjKind.WGS84))
        if self == ProjSource.GDA94:
            return pyproj.CRS(str(ProjKind.GDA94))
        if self == ProjSource.AEA_RAINFIELDS3:
            return pyproj.CRS(str(ProjKind.AEA_AUS))
        if self == ProjSource.HIMA8_HIMA9:
            return pyproj.CRS(str(ProjKind.GEO141_AUS))
        # default
        raise NotImplementedError("Projection unsupported")

    def get_crs_or_default(self, ds: xr.Dataset = None) -> pyproj.CRS:
        """
        Returns the pyproj projection object/CRS for a a particular
        projection kind (`proj_kind`).

        if `ds` is specified, this method attempts to grab the projection
        from precanned attributes.
        """
        default_proj = self.get_default_proj()
        warn_proj_defaulted = (
            f"Failed to resolve projection for {self}, using default projection instead: {str(default_proj)}"
        )

        if ds is None:
            return default_proj
        else:
            # we warn if we don't extract a projection, because if a dataset
            # is specified we expect a projection attr to be there.
            if not self.proj_attr_exists(ds):
                warnings.warn(warn_proj_defaulted)
                return default_proj

            # try to extract projections from attributes
            try:
                if self == ProjSource.AEA_RAINFIELDS3:
                    crsobj = pyproj.CRS.from_cf(ds.proj.attrs)
                    return crsobj
                elif self == ProjSource.HIMA8_HIMA9:
                    crsobj = pyproj.CRS.from_cf(ds.geostationary.attrs)
                    return crsobj
            except PYPROJ_EXCEPTION_TRAPS:
                warnings.warn(warn_proj_defaulted)
                return default_proj

            raise NotImplementedError("Attribute extraction from this proj kind is unsupported")


class CoordUnits(IntEnum):
    #: unsupported units, e.g. radians and metres are unsupported.
    UNSUPPORTED = 0

    #: angle in degrees: used for lat/lon (commonly degrees for both)
    DEGREES = 1

    #: angle in radians: for lat/lon, but not typically used
    RADIANS = 2

    #: distance in km: used for x/y (mainly radar)
    KM = 3

    #: distance in meters: used for x/y (mainly satellite)
    METRES = 4

    UNKNOWN = auto()

    @staticmethod
    def to_km(points: np.ndarray) -> np.ndarray:
        return points / KM_TO_M_MULTIPLIER


class TargetGridAus_Rectilinear:
    """
    IMPORTANT:
        This target grid is
          - in lon-lat (degrees), except for rare exceptions
          - evenly spaced in a given dimension (i.e. rectilinear)
        Source units maybe in metres or km - need to be specified so that they
        can be converted to standard units in the interp_info.
    """

    # default target grid with 0.01 degree spacing
    lon_minmax: tuple[np.float64, np.float64] = [110.0, 155.0]
    lon_count: np.float64 = 4500
    lat_minmax: tuple[np.float64, np.float64] = [-45.0, -5.0]
    lat_count: np.float64 = 4000
    units_lonlat: CoordUnits = CoordUnits.DEGREES

    @property
    def area_of_interest(self) -> AreaOfInterest:
        #: derived from target lonlat projection
        #: give extra boundary gap so that pyproj doesn't complain
        return AreaOfInterest(
            self.lon_minmax[0] - 5,  # west
            self.lat_minmax[0] - 5,  # south
            self.lon_minmax[1] + 5,  # east
            self.lat_minmax[1] + 5,  # north
        )

    @functools.cached_property
    def grid_lonlat(self) -> NumpyMeshGrid2D:
        # target grid is always rectilinear for this class
        lon_1d = np.linspace(*self.lon_minmax, num=self.lon_count, endpoint=False)
        lat_1d = np.linspace(*self.lat_minmax, num=self.lat_count, endpoint=False)

        # NOTE: order matters
        return np.meshgrid(lon_1d, lat_1d)


class SourceCRSMapper(NamedTuple):
    """
    Used for projection from a source projection to a destination projection

    The destination is actually the "source" data since typically if you're
    using this module you're already in the "projected" space. So think of
    everything in reverse.

    NOTE: typically only `projsrc_to` needs a reference `ds_ref` to derive
    its projection string. See: `get_transform_and_projpair`

    # [T2]
    # TODO: use pyproj.sync.get_transform_grid_list to get a precanned
    # transform grid list
    """

    # These are static,
    projsrc_from: ProjSource
    projsrc_to: ProjSource

    def get_transform_and_projpair(
        self,
        area_of_interest: AreaOfInterest,
        ds_ref=None,  # ds_ref => projstr_to
    ) -> CRSTransformPair:
        proj_from = self.projsrc_from.get_crs_or_default(ds=None)
        proj_to = self.projsrc_to.get_crs_or_default(ds=ds_ref)
        crspair = (proj_from, proj_to)
        transformer = SourceCRSMapper.transformer(
            proj_from,
            proj_to,
            area_of_interest,
        )
        return (crspair, transformer)

    @staticmethod
    def transformer(
        proj_from,
        proj_to,
        area_of_interest: AreaOfInterest = None,
    ) -> Transformer:
        # we always want to map geodetic crs for these
        if (not proj_to.is_geographic) or (proj_to.coordinate_system.name == "cartesian"):
            return pyproj.Proj.from_crs(
                proj_from.geodetic_crs,  # source should be geographic
                proj_to,  # destination should be cartesian
                always_xy=True,
                area_of_interest=area_of_interest,
            )
        else:
            raise ValueError(
                "PETProj.SourceCRSMapper: Failed to create forward projection. "
                "Only x-y to lon-lat is supported the forward projection must "
                "be FROM a geodetic coordinate system TO a cartesian "
                "coordinate system."
            )


@dataclass
class ProjLonLatAus_Rectilinear:
    """
    Base class for x-y to lon-lat projection. It outputs an _interpolated_
    dataset with a rectilinear lon-lat grid

    Terminology:
        source | SOURCE dataset/projection etc. in x-y

        target | TARGET dataset or output of the projection algorithm in lon-lat

        from   | TARGET dataset. "From" is a projection concept, in this case
               | from is the lon-lat coordinates always. Since our target
               | dataset is rectilinear in lon-lat, from is our TARGET. If
               | we were interested in x-y *to* would be our TARGET

        to     | Similar reason to from - this is our SOURCE dataset because
               | the input is in x-y and the coordinate transforms we deal
               | with are from lon-lat TO x-y

    NOTE: It is assumed that the source dataset is already named "x-y". This is
    because any pre-liminary stages prior to this is in-charge of variable
    munging.

    However, an option is given to rename the resultant lon-lat coordinates of
    the dataset if needed.

    IMPORTANT:
    - Only tested for forward projection (i.e. lon-lat -> x-y)
    - This is a lossy projector that compresses the target dimensions into 1-D
      coordinates to preserve even spacing.
    """

    _: KW_ONLY

    #: projection source to projection destination (required)
    crs_mapper: SourceCRSMapper

    #: reference x-y coordinates for the source dataset (required)
    units_xy: CoordUnits

    #: type of interpolation to perform (optional - default "linear")
    interp_method: str = "linear"

    #: Scipy normally tends to set the border values to NaNs, sometimes its
    #: desirable to fill them. However, by default this is not recommended.
    #: If the user sets this to true, it's their responsiblity to manage it.
    #: No guarantees are made on accuracy when this is set to True.
    extrapolate_border: bool = False

    # ------------------------------------------------------------------------
    # Defaults - do not need to be changed unelss absolutely necessary

    #: naming convention for latitude
    name_lat = "latitude"

    #: naming convention for longitude
    name_lon = "longitude"

    #: derive projection from the provided dataset
    use_default_proj: bool = False

    #: target grid
    target_grid = TargetGridAus_Rectilinear()

    def interpolate_xy_to_lonlat(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Although we're "inverting" coordinates from x-y to lon-lat. This
        operation is technically not a inverse projection. It is actually an:

        - inverse crs transform of the target rectilinear grid to match the
          source CRS (lon-lat to lon-lat), followed by

        - a *forward* projection onto x-y from the "effective" rectilinear
          grid, followed by

        - an interpolation operation (x-y to x-y) on the *source* dataset
          (`ds`) that matches the *target* rectilinear grid (lon-lat) points in
          space.

        The interpolated target x-y grid, which is likely irregularly spaced, does
        still exactly represent the target rectilinear lon-lat grid. Thus, we
        can just re-assign the data to lon-lat (1-D by 1-D) coordinates
        *independent* of x-y coordinates.

        IMPORTANT:

        Because the transformations are not necessarily linear. A linear
        interpolation in x-y is unlikely to map to a linear interpolation in
        lon-lat.

        However, as long as the resultant interpolated x-y grid is monotonic.
        The error is deterministically bounded by neighbouring grid points in
        the source x-y grid because of how interpolation algorithms typically
        work.
        """
        # ---
        # 1. map CRS to match target dataset with source dataset
        crspair, transformer = self.crs_mapper.get_transform_and_projpair(
            ds_ref=ds,
            area_of_interest=self.target_grid.area_of_interest,
        )

        # ---
        # 2. forward project to xy using the source *data* CRS
        crs_from, crs_to = crspair
        tgt_grid_lon, tgt_grid_lat = self.target_grid.grid_lonlat
        xy_grid = transformer.transform(
            tgt_grid_lon.T,
            tgt_grid_lat.T,
            direction=TransformDirection.FORWARD,
            errcheck=False,
            radians=False,  # degrees
        )

        # [T5]
        # TODO: errcheck is intentionally false, since we need to allow some
        # invalid values near boundaries, but we should check that its not all
        # "inf"

        # ---
        # 3. convert back to km if coordinate was originally in KM
        grid_x, grid_y = xy_grid
        if self.units_xy == CoordUnits.KM:
            grid_x = CoordUnits.to_km(grid_x)
            grid_y = CoordUnits.to_km(grid_y)

        # ---
        # 4. get x-y grid as a function of 1-d lon-lat target grid
        da_xproj, da_yproj = self._grid_points_to_dataarray_coords(
            grid_x,
            grid_y,
            tgt_grid_lon,
            tgt_grid_lat,
        )

        # ---
        # 5. interpolate the source, usually regularly spaced, dataset in x-y
        #    to the (irregular spaced) x-y representation of the (regular
        #    spaced) rectilinar lon-lat grid.

        # NOTE: xarray requires setting fill_value to None for extrapolation to
        # kick in in scipy
        scipy_kwargs = None
        if self.extrapolate_border:
            scipy_kwargs = {"fill_value": None}

        ds_interp = ds.interp(
            x=da_xproj,
            y=da_yproj,
            method=self.interp_method,
            kwargs=scipy_kwargs,
        )

        # NOTE: I've left the x-y values in case they are useful for the inversion
        return ds_interp

    def _grid_points_to_dataarray_coords(
        self,
        x_grid: npt.ArrayLike,
        y_grid: npt.ArrayLike,
        lon_grid: npt.ArrayLike,
        lat_grid: npt.ArrayLike,
    ) -> tuple[xr.DataArray, xr.DataArray] | None:
        """
        Helper method to create a mapped xarray dataarray tuple from xy to
        lonlat. This is actually quite tricky to get right, as the axes are not
        consistent and floating point issues may cause issues with the grids
        being recongized as rectilinear.
        """
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
            da_x = self._make_lonlat_coord_array(x_grid, lon_1d, lat_1d)
            da_y = self._make_lonlat_coord_array(y_grid, lon_1d, lat_1d)
            return (da_x, da_y)
        else:
            raise ValueError("ERROR: provided meshgrid is not rectilinear.")

    def _make_lonlat_coord_array(self, coord_grid, lon_1d, lat_1d) -> xr.DataArray:
        return xr.DataArray(
            coord_grid,
            dims=[self.name_lon, self.name_lat],
            coords={self.name_lon: lon_1d, self.name_lat: lat_1d},
        )


class Rainfields3ProjAus(ProjLonLatAus_Rectilinear):
    """
    Projection used for Radar 310 nation-wide product

    see: `ProjSource` for more details
    """

    def __init__(self):
        # This flow is FROM latlon TO xy, so from_* is our target grid if our
        # target system is latlon
        self.crs_mapper = SourceCRSMapper(
            projsrc_from=ProjSource.GDA94,
            projsrc_to=ProjSource.AEA_RAINFIELDS3,
        )
        self.units_xy = CoordUnits.KM
        self.interp_method = "linear"
        # define any custom initialisation below
        # >>>

    def __call__(self, ds: xr.Dataset):
        # <<<
        # define any custom pre processing above
        ds_interp = self.interpolate_xy_to_lonlat(ds)
        # define any custom post processing below
        # >>>
        return ds_interp


@dataclass
class HimawariProjAus(ProjLonLatAus_Rectilinear):
    """
    Common projection used for Himawari 8/9 in the AUS region

    see: `ProjSource` for more details
    """

    def __init__(self):
        self.crs_mapper = SourceCRSMapper(
            projsrc_from=ProjSource.GDA94,
            projsrc_to=ProjSource.HIMA8_HIMA9,
        )
        self.units_xy = CoordUnits.METRES
        self.interp_method = "linear"
        # define any custom initialisation below
        # >>>

    def __call__(self, ds: xr.Dataset):
        # <<<
        # define any custom pre processing above
        ds_interp = self.interpolate_xy_to_lonlat(ds)
        # define any custom post processing below
        # >>>
        return ds_interp
