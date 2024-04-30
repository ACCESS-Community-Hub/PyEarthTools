"""
National Computing Infrastructure specific Indexes

| Name        | Description |
| :---        |       ----: |
| [ERA5][edit_archive_NCI.ERA5]                | ECWMF ReAnalysis v5       |
| [ACCESS][edit_archive_NCI.ACCESS]            | Australian Community Climate and Earth-System Simulator       |
| [AGCD][edit_archive_NCI.AGCD]                | Australian Gridded Climate Data        |
| [BRAN][edit_archive_NCI.BRAN]                | Bluelink ReANalysis        |
| [OceanMaps][edit_archive_NCI.OceanMaps]      | Ocean Modelling and Analysis Prediction System        |
| [MODIS][edit_archive_NCI.MODIS]              | MODerate resolution Imaging Spectroradiometer       |
| [Himiwari][edit_archive_NCI.Himiwari]        | Himiwari 8/9 satellite data       |
| [BARRA][edit_archive_NCI.BARRA]              | Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia       |
"""

import edit.data
from edit.data.archive import register_archive


ROOT_DIRECTORIES = {
    "ACCESS": "/g/data/wr45/ops_aps3/access-{region}/1/",
    "ACCESS_S": "/g/data/ux62/access-s2/{type}/",
    "AGCD": "/g/data/zv2/agcd/v1",
    "ERA5": "/g/data/rt52/era5/{level}-levels/{resolution}/",
    "HIMIWARI": "/g/data/rv74/satellite-products/arc/der/himawari-ahi/solar/p1s/latest/",
    "BRAN": "/g/data/gb6/BRAN/BRAN2020/",
    "OceanMaps": "/g/data/rr6/OceanMAPS/",
    "MODIS": "/g/data/fj4/MODIS_LAI/{region}/nc/",
    "BARRA": "/g/data/cj37/BARRA/BARRA_{region}/{version}/{datatype}",
    "BARPA": "/g/data/ia39/australian-climate-service/release",
}


register_archive('ROOT_DIRECTORIES')(ROOT_DIRECTORIES)

import edit_archive_NCI

from edit_archive_NCI.ACCESS    import     ACCESS
from edit_archive_NCI.AGCD      import       AGCD
from edit_archive_NCI.BRAN      import       BRAN
from edit_archive_NCI.BARRA     import      BARRA
from edit_archive_NCI.BARPA     import      BARPA
from edit_archive_NCI.ERA5      import       ERA5
from edit_archive_NCI.MODIS     import      MODIS
from edit_archive_NCI.OceanMaps import  OceanMaps
from edit_archive_NCI.Himiwari  import   Himiwari

register_archive('NCI')(edit_archive_NCI)
