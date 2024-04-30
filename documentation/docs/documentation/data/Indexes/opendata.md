# ECMWF Open Store

ECMWF has recently provided access to the AIFS & IFS forecasts accessible at `https://data.ecmwf.int/forecasts/`.

To provide access, `edit` now has a `DownloadIndex` to retrieve and cache data.

This index uses `ecmwf.opendata` as the core download tool.

## Example

### AIFS

```python
import edit.data

opendata_index = edit.data.download.opendata.AIFS('msl', step = (0, 120, 6), cache = '~/AIFS_Data/')
opendata_index.latest()
```

This will retrieve `msl` from the AIFS from 0 to 120 hours at a 6 hour interval, and save it at `~/AIFS_Data/` automatically identifying the latest data available.

## Overriding

If data has been previously downloaded for a set date, but the levels or steps differ on disk to what is requested, data may not be downloaded
To fix this, simply use the `.override` context manager provided.

```python
with opendata_index.override:
    opendata_index.latest()
```
