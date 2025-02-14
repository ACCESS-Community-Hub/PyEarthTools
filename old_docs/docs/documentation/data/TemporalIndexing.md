# Temporal Indexing Behaviour

`pyearthtools` allows for more complex time based indexing with use of the [series][pyearthtools.data.AdvancedTimeIndex.series] function, however, it's fine detailed operations may not be entirely obvious.

There are three main time resolutions which need to be considered,

## Consideration

| Name / Object   | Explanation |
| --------------- | ------- |
| Data  | The data temporal resolution       |
| Start  | Specified start of the series        |
| Interval  | Resolution of the interval    |

By default `pyearthtools.data` sets the resolution of the start time to at least the resolution of the interval, which if not given
defaults to the data's interval.

Consider:

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021', '2021-03', interval = (1, 'hour'))
```

The three resolutions mentioned above have the following values,
| Name / Object   | Value |
| --------------- | ------- |
| Data        | Hourly  |
| Start       | Year |
| Interval    | Hour |

The expected behaviour of the above call, is all data in Jan and Feb of 2021, which `pyearthtools` follows.

Ultimately, `pyearthtools` will respect the resolution of the given `start`, and step by the given interval, retrieving all the data 
which are within the specified resolution at each step. 

!!! Note start.resolution == interval.resolution but > data.resolution
    If the start and interval have the same resolution, and the interval is 1, all data will be retrieved at the data resolution
    effectivaly negating the point of the interval. But if not 1, the step takes effect.

If the interval is monthly, the start daily and the data hourly, all hours of data on that day each month will be returned.

## Examples

Interval: month, start: daily, data: hourly

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021-01-13', '2021-06', interval = (1, 'month')).time

##  All hourly data on the 13th of 01-05 months

```

Interval: month, start: hourly, data: hourly

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021-02-03T14', '2021-06', interval = (1, 'month')).time

##  All data at 14:00 on 3rd of each month between 02-05 

```

Interval: day, start: hourly, data: hourly

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021-02-03T14', '2021-06', interval = (1, 'day')).time
##  All data at 14:00 evert day between 02-05 months starting on the 3rd

```

Interval: day, start: hourly, data: hourly
with end: day

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021-02-03T14', '2021-06-06', interval = (1, 'day')).time
##  All data at 14:00 evert day between 02-05 months starting on the 3rd, ending on 6th

```

Interval: month, start: month, data: hourly

```py
from pyearthtools.data.archive import ERA5
era5 = ERA5('2t', level = 'single')

era5.series('2021-02', '2021-06', interval = (1, 'month')).time

##  All hourly data on the between 02-05 months

```

## Usage if not handled

If the exact data retrieval you are after is not handled here, it is possible to use these operations as atomic steps and
then merge the data together later.
