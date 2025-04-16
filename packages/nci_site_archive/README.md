# `edit.archive`'s for the National Computing Infrastructure

This package contains the `edit.Index`'s for the NCI, which consists of

| Name        | Description |
| :---        |       ----: |
| ERA5                | ECWMF ReAnalysis v5       |
| ACCESS              | Australian Community Climate and Earth-System Simulator       |
| AGCD                | Australian Gridded Climate Data        |
| BRAN                | Bluelink ReANalysis        |
| OceanMaps           | Ocean Modelling and Analysis Prediction System        |
| MODIS               | MODerate resolution Imaging Spectroradiometer       |
| Himawari            | Himawari 8/9 satellite data       |
| BARRA               | Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia       |

## Installation

This should be installed as a package as with the other PyEarthTools sub-packages.

## Usage

If installed, this package will automatically be imported when `pyearthtools.data` is if the user is on NCI.

It can also be explicitly imported if installed by,

```py
import site_archive_nci
```

Once imported all archives are accesible underneath `pyearthtools.data.archive`

```python
import pyearthtools.data

BRAN_data = pyearthtools.data.archive.BRAN('ocean_temp', resolution = 'daily')
BRAN_data

# BRAN
#         Description                    Bluelink ReANalysis
#                  range                          1993-current
#         Initialisation                 
#                  resolution                     daily
#                  transforms                     {}
#                  depth_value                    None
#                  variables                      ocean_temp
#         Transforms                     
#                  ConformNaming                  {'latitude': ['lat', 'Latitude', 'yt_ocean', 'yt'], 'longitude': ['lon', 'Longitude', 'xt_ocean', 'xt'], 'time': ['Time']}
#                  StandardLongitude180180        {'type': '-180-180'}
#                  VariableTrim                   {'variables': ['temp']}

```

However, as this is a registered archive within `edit.data`, `edit_archive_NCI` doesn't need to be manually imported.
