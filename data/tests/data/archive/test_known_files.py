# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from pathlib import Path

import pytest

import edit.data


@pytest.mark.parametrize("variable,domain", [("sfc/sfc_temp", "g"), ("sfc/mslp", "g")])
def test_ACCESS_analysis(variable, domain):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.ACCESS.analysis(variable, domain).exists("2021-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable,domain", [("sfc/sfc_temp", "g"), ("sfc/mslp", "g")])
def test_ACCESS_forecast(variable, domain):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.ACCESS.forecast(variable, domain).exists("2021-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable,level", [("tcwv", "single"), ("cape", "single")])
def test_ERA5(variable, level):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.ERA5(variable, level=level).exists("2021-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable", [("cloud_optical_depth"), (None)])
def test_Himiwari(variable):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.Himiwari(variable).exists("2021-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable,type", [("ocean_eta_t", "month"), ("ocean_salt", "month")])
def test_BRAN(variable, type):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.BRAN(variable, type).exists("2021-01")
    else:
        assert False


@pytest.mark.parametrize(
    "variable,type, subvar",
    [("eta", "analysis", "ocean_an00"), ("salt", "analysis", "ocean_an00")],
)
def test_OceanMaps(variable, type, subvar):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.OceanMaps(variable, type, subvar).exists("2022-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable,region,resolution", [("lai", "AU", "8-daily")])
def test_MODIS(variable, region, resolution):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.MODIS(variable, region, resolution=resolution).exists("2021-01-01")
    else:
        assert False


@pytest.mark.parametrize("variable,region,datatype", [("slv/soil_temp", "R", "analysis")])
def test_BARRA(variable, region, datatype):
    if hasattr(edit.data.archive, "NCI"):
        assert edit.data.archive.BARRA(variable, region=region, datatype=datatype).exists("2000-01-01")

    else:
        assert False
