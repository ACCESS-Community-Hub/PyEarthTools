import edit
import edit.data.archive
import era5lowres

var = var=['u', 'v']
UandV = edit.data.archive.era5lowres(var)
data = UandV['1984-01-01']

data
