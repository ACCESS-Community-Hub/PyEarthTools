import pyearthtools
import pyearthtools.data.archive
import era5lowres

var = var=['u', 'v']
UandV = pyearthtools.data.archive.era5lowres(var)
data = UandV['1984-01-01']

data
