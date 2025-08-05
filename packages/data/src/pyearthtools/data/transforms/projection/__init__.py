from dataclasses import dataclass  # , field, KW_ONL
import xarray as xr

from . import _projection_manager as projmanager

from pyearthtools.data.transforms import Transform


class XYtoLonLatRectilinear(Transform):
    """
    Projection class that transforms datasets from a (usually rectilinear)
    x-y to lon-lat (always rectilinear).

    !Caution - only tested on one specific projection and region

    In most cases if x-y is rectilinear it is unlikely that lon-lat is
    regularly spaced. So this class will do interpolation to conform to a
    rectilinear grid.

    The petproj module has some template classes for different products that
    can be used to initialise this transform.
    """

    def __init__(self, projection_method: projmanager.ProjLonLatAus_Rectilinear):
        # default initialiser
        self._inner_proj = projection_method

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        if not isinstance(ds, xr.Dataset):
            raise NotImplementedError(
                """Only xr.Dataset is supported currently as it contains metadata that is
                extracted for projection, you can force a default projection using custom
                options - but this is not in the scope of this tutorial currently."""
            )
        ds_ret = self._inner_proj(ds)
        return ds_ret


class Rainfields3ProjAus(projmanager.ProjLonLatAus_Rectilinear):
    """
    Projection used for Radar 310 nation-wide product

    see: `ProjSource` for more details
    """

    def __init__(self):
        # This flow is FROM latlon TO xy, so from_* is our target grid if our
        # target system is latlon
        self.crs_mapper = projmanager.SourceCRSMapper(
            projsrc_from=projmanager.ProjSource.GDA94,
            projsrc_to=projmanager.ProjSource.AEA_RAINFIELDS3,
        )
        self.units_xy = projmanager.CoordUnits.KM
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
class HimawariProjAus(projmanager.ProjLonLatAus_Rectilinear):
    """
    Common projection used for Himawari 8/9 in the AUS region

    see: `ProjSource` for more details
    """

    def __init__(self):
        self.crs_mapper = projmanager.SourceCRSMapper(
            projsrc_from=projmanager.ProjSource.GDA94,
            projsrc_to=projmanager.ProjSource.HIMA8_HIMA9,
        )
        self.units_xy = projmanager.CoordUnits.METRES
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


__all__ = ["Rainfields3ProjAus", "HimawariProjAus"]
