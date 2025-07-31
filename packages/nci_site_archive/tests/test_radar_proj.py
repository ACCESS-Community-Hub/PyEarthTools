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
Test suite for RadarProj (currently coupled with _Rainfields3.py)
"""

import functools

import numpy as np
import pyproj
import pytest
import platform
import xarray as xr

from site_archive_nci._Rainfields3 import (
    ErrorRadarProj,
    ProjErrorStatus,
    ProjKind,
    RadarProj,
    WarnRadarProj,
)

PYPROJ_SAMPLE = pyproj.Proj("+proj=aea +lat_1=-36 +lat_2=-18 +lon_0=132 +units=m")
EXPECTED_KEYS = [
    "grid_mapping_name",
    "standard_parallel",
    "latitude_of_projection_origin",
    "longitude_of_prime_meridian",
    "false_easting",
    "false_northing",
    "longitude_of_central_meridian",
]

PROJ_CF = PYPROJ_SAMPLE.crs.to_cf()
DA_PROJ_ATTR = {k: PROJ_CF[k] for k in PROJ_CF if k in EXPECTED_KEYS}
DA_PROJ_VAR = "proj"
DA_PROJ_VAL = 0

# --------------
# sample dataset
# --------------
# 10km x 10km grid over 6x6 points
#
# [
#   [0, 1, 2, 3, 4, 5],    # row 1
#   [6, 7, 8, 9, 10, 11],  # row 2
#   ...,                   # ...
#   [..., 35],             # row 6
# ]
#
# since projection origin is lat=0deg, but lat_parallels=(lat1=-36,lat2=-18),
# 0 is not contained in this so need to ffset.
#
# arbitrary offset: y=2000km - inferred from radar data for this proj.
DS_TEST = xr.Dataset(
    data_vars={
        "temp": xr.DataArray(np.reshape(np.arange(0, 36), (6, 6)), dims=["x", "y"]),
        DA_PROJ_VAR: xr.DataArray(DA_PROJ_VAL, attrs=DA_PROJ_ATTR),
    },
    coords={
        "x": np.array([-5, -3, -1, 1, 3, 5]),
        "y": np.array([-5, -3, -1, 1, 3, 5]) - 2000,
    },
)
# projection of [[-5, -5-2000], [5, 5-2000]] (in km)
APPROX_LATLON_BOUNDS = PYPROJ_SAMPLE(
    [-5 * 1000, 5 * 1000],
    [-2005 * 1000, -1995 * 1000],
    inverse=True,
    errcheck=True,
    radians=False,
)
LATLON_BOUNDS_TOO_FAR = PYPROJ_SAMPLE(
    [-20 * 1000, 20 * 1000],
    [-2035 * 1000, -1975 * 1000],
    inverse=True,
    errcheck=True,
    radians=False,
)


class TestRadarProj:
    # --- test success states ---

    def test_pyproj_parsing(self):
        """
        Test pyproj attr can be extracted from dataset
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        assert RadarProj.REQUIRED_PROJATTR_GRIDMAPPINGNAME in pyprojattr

    def test_getprojkind(self):
        """
        Test projkind is mapped correctly;
        currently only ALBERS_CONICAL_EQUAL_AREA is supported.
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projkind = RadarProj._get_projkind_from_projattr(pyprojattr)
        assert projkind == ProjKind.ALBERS_CONICAL_EQUAL_AREA

    def test_pyprojobj_convert(self):
        """
        Test conversion works to pyproj object doesn't spit any errors.
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projobj, crsobj = RadarProj._transform_projattr_to_pyprojobj(pyprojattr)
        assert isinstance(projobj, pyproj.Proj)
        assert isinstance(crsobj, pyproj.CRS)
        assert "aea" in projobj.srs

    def test_warn_inconsistent_proj_mut(self):
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projobj, crsobj = RadarProj._transform_projattr_to_pyprojobj(pyprojattr)
        proj_cache = []

        print("check proj A (1) - NO_WARN:")
        print(projobj.definition_string())
        print(proj_cache)
        RadarProj._warn_inconsistent_proj_mut(projobj, proj_cache, do_warn=True)
        assert len(proj_cache) == 1

        # --- Case 1: Proj A repeated - no warn ----
        # shouldn't warn - same proj
        print("check proj A (2) - NO_WARN:")
        print(projobj.definition_string())
        RadarProj._warn_inconsistent_proj_mut(projobj, proj_cache, do_warn=True)
        # length shoudn't change
        assert len(proj_cache) == 1

        # --- Case 2: New - Proj B - warn ----
        # new proj
        with pytest.warns(WarnRadarProj):
            proj_new = pyproj.Proj("EPSG:4326")
            print("check proj B (1) - WARN:")
            print(proj_new.definition_string())
            # _should_ warn
            RadarProj._warn_inconsistent_proj_mut(proj_new, proj_cache, do_warn=True)
            # length should be 2 now
            assert len(proj_cache) == 2

        # --- Case 3: Proj B again - no warn ----
        # Proj B again - no warn
        print("check proj B (2) - NO_WARN:")
        print(proj_new.definition_string())
        RadarProj._warn_inconsistent_proj_mut(proj_new, proj_cache, do_warn=True)
        assert len(proj_cache) == 2

        # --- Case 4: Proj A' - slightly modified A - no warn ---
        #     => lat_2=0
        #     => do_warn=False
        #     => no warn, but length still increases
        print("check proj A' (1) - NO_WARN:")
        proj_newish = pyproj.Proj("+proj=aea +lat_1=-36 +lat_2=0 +lon_0=132 +units=m")
        RadarProj._warn_inconsistent_proj_mut(proj_newish, proj_cache, do_warn=False)
        assert len(proj_cache) == 3

        # --- Case 5: Proj A'' - slightly modified A' - warn ---
        #     => lon_0=125
        #     => do_warn=False
        #     => no warn, but length still increases
        with pytest.warns(WarnRadarProj):
            print("check proj A'' (1) - WARN:")
            proj_newish = pyproj.Proj("+proj=aea +lat_1=-36 +lat_2=0 +lon_0=125 +units=m")
            RadarProj._warn_inconsistent_proj_mut(proj_newish, proj_cache, do_warn=True)
            assert len(proj_cache) == 4

    def test_inverse_lonlat_grid(self):
        """
        Test inversion flow: lat-lon coord assignment via x-y proj inversion
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projobj, _ = RadarProj._transform_projattr_to_pyprojobj(pyprojattr)
        ds_inv, _, _ = RadarProj._map_xy_meshgrid_to_lonlat_grid(DS_TEST, projobj)
        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS

        # no change in values
        assert np.all(ds_inv.temp.values == DS_TEST.temp.values)
        # coords assigned
        assert getattr(ds_inv, "lon", None) is not None
        assert getattr(ds_inv, "lat", None) is not None
        # expect 2-D grid
        assert ds_inv.lon.shape == (6, 6)
        assert ds_inv.lat.shape == (6, 6)
        # projections within expected range
        assert np.min(ds_inv.lat) >= -36
        assert np.max(ds_inv.lat) <= -18
        # distortion cannot be more than 1 degree close to the center
        assert np.min(ds_inv.lon) >= lon0 - 1
        assert np.max(ds_inv.lon) <= lon1 + 1

    @pytest.mark.skipif(platform.system() == "Darwin", reason="This specific test fails on macOS")
    @pytest.mark.parametrize(
        "interp_method",
        ["linear", "slinear", "cubic"],
    )
    def test_xy_to_latlon_interp(self, interp_method):
        """
        Test interpolation flow: x-y interpolation from lat-lon grid
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projobj, _ = RadarProj._transform_projattr_to_pyprojobj(pyprojattr)
        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS

        # Note: including endpoint, since min/max which are used to compute
        # lon0/lon1/lat0/lat1 return elements within the extents (not outside).
        _fn_mesh_lonlat = functools.partial(
            RadarProj.make_lonlat_meshgrid,
            lon_extent=(lon0, lon1),
            lat_extent=(lat0, lat1),
            endpoint=True,
        )

        _fn_interp = functools.partial(
            RadarProj._interp_xy_grid_from_lonlat_meshgrid,
            ds=DS_TEST,
            pyprojobj=projobj,
            interp_method=interp_method,
        )

        def _common_asserts(_ds, num_lon, num_lat):
            # extrapolation should have kicked in since we are just using the
            # inverse projection on the lat-lon boundaries to forward project
            # again. So we shouldn't expect NaNs
            assert np.sum(np.isnan(_ds.temp.values)) == 0

            if num_lon == DS_TEST.temp.shape[0] and num_lat == DS_TEST.temp.shape[1]:
                # all values should be similar if the shape is similar
                assert np.mean(np.abs(DS_TEST.temp.values - ds_interp.temp.values)) <= 1
            else:
                # Otherwise we are either upsampling or downsampling
                if num_lon <= DS_TEST.temp.shape[0] or num_lat <= DS_TEST.temp.shape[1]:
                    # downsampling => relax requirements, but still should be within bounds
                    assert np.abs(_ds.temp.mean() - DS_TEST.temp.mean()) <= DS_TEST.temp.std()
                    # max should be closer to DS_TEST.max than DS_TEST.mean
                    assert _ds.temp.max() > (DS_TEST.temp.max() - DS_TEST.temp.mean()) * 0.5
                    # min should be closer to DS_TEST.min than DS_TEST.mean
                    assert _ds.temp.min() < (DS_TEST.temp.mean() - DS_TEST.temp.min()) * 0.5
                else:
                    # upsampling => should still be roughly close since values should be
                    # stable within the data boundary.
                    assert np.abs(_ds.temp.mean() - DS_TEST.temp.mean()) <= 1
                    assert np.abs(_ds.temp.min() - DS_TEST.temp.min()) <= 1
                    assert np.abs(_ds.temp.max() - DS_TEST.temp.max()) <= 1

            # coords assgined
            assert getattr(_ds, "lon", None) is not None
            assert getattr(_ds, "lat", None) is not None
            # coords match (almost) exactly in this case
            np.testing.assert_approx_equal(np.nanmin(_ds.lon), lon0)
            np.testing.assert_approx_equal(np.nanmax(_ds.lon), lon1)
            np.testing.assert_approx_equal(np.nanmin(_ds.lat), lat0)
            np.testing.assert_approx_equal(np.nanmax(_ds.lat), lat1)
            # check shapes
            assert _ds.lon.shape == (num_lon,)
            assert _ds.lat.shape == (num_lat,)
            assert _ds.temp.shape == (num_lon, num_lat)

        # interp: 6 by 6
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lat=6, num_lon=6))
        _common_asserts(ds_interp, 6, 6)

        # downsample: 3 by 3
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=3, num_lat=3))
        _common_asserts(ds_interp, 3, 3)

        # upsample: 12 by 12
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=12, num_lat=12))
        _common_asserts(ds_interp, 12, 12)

        # non-square: 9 by 17
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=9, num_lat=17))
        _common_asserts(ds_interp, 9, 17)

        # non-square-flipped: 18 by 10
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=18, num_lat=10))
        _common_asserts(ds_interp, 18, 10)

    def test_xy_to_latlon_interp_no_extrap(self):
        """
        Test: no extrapolation scenarios
            * boundary > step size * 2 => boundary is nan filled
            * boundary is okay, but extrapolation not supported => either nan filled or error
        """
        pyprojattr = RadarProj._extract_projattr_from_ds(DS_TEST)
        projobj, _ = RadarProj._transform_projattr_to_pyprojobj(pyprojattr)

        # Note: including endpoint, since min/max which are used to compute
        # lon0/lon1/lat0/lat1 return elements within the extents (not outside).

        # ---------------------------------------------
        #  Case 1: method supported but latlon too far
        # ---------------------------------------------
        (lon0, lon1), (lat0, lat1) = LATLON_BOUNDS_TOO_FAR
        method = "linear"
        _fn_mesh_lonlat = functools.partial(
            RadarProj.make_lonlat_meshgrid,
            lon_extent=(lon0, lon1),
            lat_extent=(lat0, lat1),
            endpoint=True,
        )
        _fn_interp = functools.partial(
            RadarProj._interp_xy_grid_from_lonlat_meshgrid,
            ds=DS_TEST,
            pyprojobj=projobj,
            interp_method=method,
        )
        ds_interp, _, _ = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=6, num_lat=6))
        # check shape retained
        assert ds_interp.temp.shape == DS_TEST.temp.shape

        # check nans inserted at boundaries
        assert all(np.isnan(ds_interp.temp.values[0, :]))
        assert all(np.isnan(ds_interp.temp.values[:, 0]))

        # check bounded
        assert ds_interp.temp.min() >= DS_TEST.temp.min() - 10
        assert ds_interp.temp.max() <= DS_TEST.temp.max() + 10

        # check still within 1 std since mean shouldn't changed too much
        assert np.nanmean(ds_interp.temp.values) - DS_TEST.temp.mean() <= DS_TEST.temp.std()

        # --------------------------------------------
        #  Case 2: latlon okay but method unsupported
        # --------------------------------------------
        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS
        # FUTURE WARNING: this behaviour is offloaded to scipy and this test
        # _may_ (but unlikely) need to be updated accordingly - see its docs.
        method = "splinef2d"
        _fn_mesh_lonlat = functools.partial(
            RadarProj.make_lonlat_meshgrid,
            lon_extent=(lon0, lon1),
            lat_extent=(lat0, lat1),
            endpoint=True,
        )
        _fn_interp = functools.partial(
            RadarProj._interp_xy_grid_from_lonlat_meshgrid,
            ds=DS_TEST,
            pyprojobj=projobj,
            interp_method=method,
        )
        ret = _fn_interp(lonlat_meshgrid=_fn_mesh_lonlat(num_lon=6, num_lat=6))
        assert ret is None

    # --- test success state: external calls ---
    # NOTE: these are rudimentary tests that only tests that the input/output is as
    # expected, they do not verify internal state. Other tests do this by unit testing
    # internal functions

    def test_xy_to_lonlat_defaults(self):
        """
        Test the most basic call, no kwargs
        """
        ds = RadarProj.xy_to_lonlat(DS_TEST)

        # basic data has not changed
        assert np.sum(ds.temp.values - DS_TEST.temp.values) < 1e-6
        assert np.all(ds.x == DS_TEST.x)
        assert np.all(ds.y == DS_TEST.y)

        # lon-lat has been added
        assert getattr(ds, "lon") is not None
        assert getattr(ds, "lat") is not None

        # lon-lat is 6x6
        assert ds.lon.shape == (6, 6)
        assert ds.lat.shape == (6, 6)

        # reprojecting gives back roughly the same x-y
        xgrid, ygrid = PYPROJ_SAMPLE(ds.lon, ds.lat, inverse=False, errcheck=True, radians=False)
        xgrid = xgrid / 1000  # convert to km
        ygrid = ygrid / 1000  # convert to km
        # unique sorts
        # fix annoying floating point stuff that make comparisons hard
        approx_unique = lambda _v: np.unique(np.round(_v * 1e6) // 1e6)  # noqa
        assert np.sum(approx_unique(xgrid.flatten()) - approx_unique(ds.x)) <= 1e-6
        assert np.sum(approx_unique(ygrid.flatten()) - approx_unique(ds.y)) <= 1e-6

    def test_xy_to_lonlat_projcache(self):
        """
        Test with cache
        """
        lcache = []
        ds_different_proj = DS_TEST.copy(deep=True)
        ds_different_proj.proj.attrs.update({"standard_parallel": (-36, 0)})
        RadarProj.xy_to_lonlat(DS_TEST, proj_cache=lcache)
        # cache length updated, but no warning
        assert len(lcache) == 1
        # no warning
        RadarProj.xy_to_lonlat(ds_different_proj, proj_cache=lcache)
        # cache length updated
        assert len(lcache) == 2

    def test_xy_to_lonlat_projcache_warn(self):
        """
        Test with cache and warning on different proj
        """
        lcache = []
        ds_different_proj = DS_TEST.copy(deep=True)
        ds_different_proj.proj.attrs.update({"standard_parallel": (-36, 0)})
        RadarProj.xy_to_lonlat(DS_TEST, proj_cache=lcache, warn_on_different_proj=True)
        # cache length updated, but no warning
        assert len(lcache) == 1
        # should warn this time
        with pytest.warns(WarnRadarProj):
            RadarProj.xy_to_lonlat(ds_different_proj, proj_cache=lcache, warn_on_different_proj=True)

        assert len(lcache) == 2

    def test_xy_to_lonlat_projcache_forceproj(self):
        """
        Test force proj - should ignore proj_cache and warn_on_different_proj
        """
        lcache = []
        ds_different_proj = DS_TEST.copy(deep=True)
        ds_different_proj.proj.attrs.update({"standard_parallel": (-36, 0)})
        force_proj = pyproj.Proj("+proj=aea +lat_1=-36 +lat_2=0 +lon_0=132 +units=m")
        # cache should not be used or updated, therefore empty
        assert len(lcache) == 0
        RadarProj.xy_to_lonlat(
            DS_TEST,
            force_proj=force_proj,
            proj_cache=lcache,
            warn_on_different_proj=True,
        )
        # force => should not actually warn even if warn_on_different_proj=True
        RadarProj.xy_to_lonlat(
            ds_different_proj,
            force_proj=force_proj,
            proj_cache=lcache,
            warn_on_different_proj=True,
        )
        # cache should still be empty
        assert len(lcache) == 0

    def test_xy_to_lonlat_projcache_interp(self):
        """
        Test that interpolate arguments are respected.

        NOTE: interp does not depend on force_proj/proj_cache etc. so it can be tested
              independently.
        """
        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS
        mg = RadarProj.make_lonlat_meshgrid(
            lon_extent=(lon0, lon1),
            num_lon=6,
            lat_extent=(lat0, lat1),
            num_lat=6,
            endpoint=True,
        )
        ds = RadarProj.xy_to_lonlat(DS_TEST, interp_lonlat=True, lonlat_meshgrid=mg)
        # lon-lat is 1D len=6
        assert ds.lon.shape == (6,)
        assert ds.lat.shape == (6,)
        # lon-lat is 1D grid with even spacing => x-y has to be a 2D grid since the
        # projection is not linear.
        assert ds.x.shape == (6, 6)
        assert ds.y.shape == (6, 6)
        # check that the data is still roughly equal
        assert np.all(ds.temp.values - DS_TEST.temp.values <= 1)
        # no nans by default
        assert not np.any(np.isnan(ds.temp.values))

    def test_make_lonlat_meshgrid(self):
        """
        Test make lon-lat meshgrid has expected dims and values
        """
        # exclude endpoint
        lons, lats = RadarProj.make_lonlat_meshgrid(
            lon_extent=(0, 6),
            num_lon=6,
            lat_extent=(0, 12),
            num_lat=12,
            endpoint=False,
        )

        assert np.max(lons) == 5
        assert np.min(lons) == 0
        assert np.max(lats) == 11
        assert np.min(lats) == 0

        # include endpoint; also flip sampling
        lons, lats = RadarProj.make_lonlat_meshgrid(
            lon_extent=(0, 12),
            num_lon=13,
            lat_extent=(0, 6),
            num_lat=7,
            endpoint=True,
        )

        assert np.max(lons) == 12
        assert np.min(lons) == 0
        assert np.max(lats) == 6
        assert np.min(lats) == 0

    # --- test failure states ---

    def test_error_attr_missing(self):
        """
        Test proj missing or grid_mapping_name not present raises error (required to
        determine projection type). force_proj must not be True in this test.
        """
        ds_noproj = DS_TEST.copy(deep=True)
        del ds_noproj["proj"]
        with pytest.raises(ErrorRadarProj, match="PES-101"):
            RadarProj.xy_to_lonlat(ds_noproj)

        # grid mapping not present
        ds_nogridmap = DS_TEST.copy(deep=True)
        del ds_nogridmap.proj.attrs["grid_mapping_name"]
        with pytest.raises(ErrorRadarProj, match="PES-101"):
            RadarProj.xy_to_lonlat(ds_nogridmap)

    def test_error_proj_unsupported(self):
        """
        Test unsupported projection. Uses a valid but unsupported projection. This test
        may need updating if the supplied projection becomes supported; intentionally
        done to test for regression.
        """
        # update this if it is now supported:
        proj_unsupported = pyproj.Proj("EPSG:4326")

        ds_projnotsup = DS_TEST.copy(deep=True)
        ds_projnotsup.proj.attrs = proj_unsupported.crs.to_cf()
        with pytest.raises(ErrorRadarProj, match="PES-102"):
            RadarProj.xy_to_lonlat(ds_projnotsup)

    def test_error_pyprojobj_conversion(self):
        """
        Test conversion from dataset attributes to pyproj.Proj object. Highlights any
        issues with incompatibility. Should raise an error if the dataset attribute is
        malformed.

        The projection method itself should be supported but the attributes themselves
        should be corrupted for this test.
        """
        ds_badprojattr = DS_TEST.copy(deep=True)
        # this should be a float pair, but pyproj.Proj is fairly fuzzy so we set it to
        # something that is meaningless for this context - hence potato
        ds_badprojattr.proj.attrs["standard_parallel"] = "potato"
        with pytest.raises(ErrorRadarProj, match="PES-103"):
            _ds_out = RadarProj.xy_to_lonlat(ds_badprojattr)
            _projobj = RadarProj._transform_projattr_to_pyprojobj(ds_badprojattr.proj.attrs)

        # try deleting the whole thing
        del ds_badprojattr.proj.attrs["standard_parallel"]
        with pytest.raises(ErrorRadarProj, match="PES-103"):
            _ds_out = RadarProj.xy_to_lonlat(ds_badprojattr)
            _projobj = RadarProj._transform_projattr_to_pyprojobj(ds_badprojattr.proj.attrs)

    # FIXME
    # TODO: all error states
    def test_error_xy_inverse_proj_failed(self, monkeypatch):
        """
        This needs to be tested by bypassing xy_to_lonlat, because we need to use a
        projection without a valid inverse.
        """
        # https://proj.org/en/stable/operations/projections/adams_hemi.html
        # adams hemisphere in a square is forward-only
        pyprojobj = pyproj.Proj("+proj=adams_hemi")
        res = RadarProj._map_xy_meshgrid_to_lonlat_grid(DS_TEST, pyprojobj)
        assert res is None

        # to test it with xy_to_lonlat, we need to monkeypatch
        # _get_projkind_from_projattr to a supported projection
        monkeypatch.setattr(
            RadarProj,
            "_get_projkind_from_projattr",
            lambda _: ProjKind.ALBERS_CONICAL_EQUAL_AREA,
        )
        with pytest.raises(ErrorRadarProj, match="PES-104"):
            res = RadarProj.xy_to_lonlat(DS_TEST, force_proj=pyprojobj)

    def test_error_status_runtime(self):
        """
        Tests that unexpected runtime error can be triggered. This is actually a
        paradoxical test that is hard to implement.

        If we were able to design a test for this then, that implies there is an issue
        with the code that needs fixing - which would then invalidate this test.

        Instead, this basically tests that the error handler raises the error without
        any functional change to the underlying code.

        Every other test is basically a guard against runtime errors so this in theory
        isn't needed other than coverage.
        """
        with pytest.raises(ErrorRadarProj, match="PES-999"):
            RadarProj._handle_error_state(ProjErrorStatus.UNEXPECTED_RUNTIME_ERROR)

    def test_no_unexpected_status(self):
        """
        Loops through all error states asserting that an error is raised.

        If it isn't then it is not handled.
        """
        for err_state in ProjErrorStatus:
            with pytest.raises(ErrorRadarProj):
                RadarProj._handle_error_state(err_state)

    def test_error_xy_to_latlon_unsupported_method(self):
        """
        Tests for unsupported method; also implies that interp_method gets propagated
        properly, since scipy is the one that will trigger this error.
        """
        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS
        mg = RadarProj.make_lonlat_meshgrid(
            lon_extent=(lon0, lon1),
            num_lon=6,
            lat_extent=(lat0, lat1),
            num_lat=6,
            endpoint=True,
        )
        with pytest.raises(ErrorRadarProj, match="PES-105"):
            _ds = RadarProj.xy_to_lonlat(DS_TEST, interp_lonlat=True, interp_method="potato", lonlat_meshgrid=mg)

    def test_xy_to_lonlat_projcache_interp_bad_args(self):
        """
        Test that lonlat grid is provided if interpolating, otherwise interpolation does
        not know what to do.
        """
        with pytest.raises(ValueError, match="lonlat_meshgrid must be specified"):
            _ds = RadarProj.xy_to_lonlat(DS_TEST, interp_lonlat=True)

    def test_xy_to_lonlat_projcache_interp_no_scipy(self, monkeypatch):
        """
        Test guard against scipy dependency not existing
        """
        # trigger import if not already
        import site_archive_nci._Rainfields3

        assert site_archive_nci._Rainfields3 is not None

        (lon0, lon1), (lat0, lat1) = APPROX_LATLON_BOUNDS
        # Note: including endpoint, since min/max which are used to compute
        # lon0/lon1/lat0/lat1 return elements within the extents (not outside).
        _fn_mesh_lonlat = functools.partial(
            RadarProj.make_lonlat_meshgrid,
            lon_extent=(lon0, lon1),
            lat_extent=(lat0, lat1),
            endpoint=True,
        )
