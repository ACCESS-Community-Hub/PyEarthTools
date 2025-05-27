import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe
import site_archive_nci

cmip5_model1 = petdata.archive.CMIP5(
    institutions="BCC", scenarios=["rcp60"], models=["bcc-csm1-1"], interval="mon", variables="tas"
)

ds_cmip_2010 = cmip5_model1["2010-01"]  # Query data along primary dimension
