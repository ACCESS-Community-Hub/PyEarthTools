# AWS Data Accessors

This package is for providing accessors for accessing data stored in public S3 buckets on AWS. 

The indices provided through the site archive load the data directly from AWS S3 buckets, so loading of data will be restricted by network speed if accessing outside AWS. It would be recommnded to consider using a [Cache operator](https://pyearthtools.readthedocs.io/en/latest/api/pipeline/pipeline_api.html#pyearthtools.pipeline.modifications.Cache) to store a local copy of the normalised data to speed up loading of data as part of a machine learning training pipeline.

### Usage
The following is a python snippet that demonstrates how to access Met Office UKV data on AWS through the site archive.

```
import datetime
import site_archive_aws
import pyearthtools.data.archive

var_list = [
    'temperature_on_pressure_levels',
    'relative_humidity_on_pressure_levels',
    'wind_speed_on_pressure_levels',
    'wind_direction_on_pressure_levels',
]

ukv_accessor = pyearthtools.data.archive.MOUKV(var_list)

sample_time = datetime.datetime(2026,3,1,12,0)

sample_time = datetime.datetime(2026,3,1,12,0)
```