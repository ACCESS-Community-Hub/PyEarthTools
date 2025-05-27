import pytest

from pyearthtools.data.download import arcoera5


def _load_sample(variables, levels, sample_time):
    arco = arcoera5.ARCOERA5(variables, levels=levels)
    sample = arco.get(sample_time)

    assert "time" in sample.coords
    assert sample.time.shape == ()

    assert "latitude" in sample.dims
    assert sample.sizes["latitude"] == 721
    assert "longitude" in sample.dims
    assert sample.sizes["longitude"] == 1440

    return sample


@pytest.mark.parametrize(
    "variables,levels,sample_time",
    [
        (["sub_surface_runoff", "peak_wave_period", "snowfall"], None, "20121201"),
        (["normalized_stress_into_ocean", "runoff"], [850, 70], "20360112T05"),
        pytest.param(None, None, "19000312", marks=pytest.mark.slow)
    ]
)
def test_load_vars(variables, levels, sample_time):
    sample = _load_sample(variables, levels, sample_time)

    if variables is None:
        variables = tuple(arcoera5.LONG_NAMES)
    assert all(varname in sample.data_vars for varname in variables)
    assert len(sample.data_vars) == len(variables)

    if levels is None:
        levels = arcoera5.LEVELS
    assert "level" in sample.dims
    assert sample.sizes["level"] == len(levels)
    assert all(level in sample.level for level in levels)


@pytest.mark.parametrize(
    "variable,levels,sample_time",
    [
        ("sub_surface_runoff", [850], "19720201"),
        ("100m_u_component_of_wind", None, "20360112T05"),
    ]
)
def test_load_1var(variable, levels, sample_time):
    sample = _load_sample(variable, levels, sample_time)

    assert variable in sample.data_vars
    assert len(sample.data_vars) == 1

    if levels is None:
        levels = arcoera5.LEVELS
    assert "level" in sample.dims
    assert sample.sizes["level"] == len(levels)
    assert all(level in sample.level for level in levels)


@pytest.mark.parametrize(
    "variables,level,sample_time",
    [
        pytest.param(None, 850, "19920327T09", marks=pytest.mark.slow),
        (["lake_depth", "land_sea_mask"], 1, "20110706"),
    ]
)
def test_load_1level(variables, level, sample_time):
    sample = _load_sample(variables, level, sample_time)

    if variables is None:
        variables = tuple(arcoera5.LONG_NAMES)
    assert all(varname in sample.data_vars for varname in variables)
    assert len(sample.data_vars) == len(variables)

    assert "level" in sample.coords
    assert sample.level.shape == ()
    assert level in sample.level


def test_renamed_vars():
    pass
