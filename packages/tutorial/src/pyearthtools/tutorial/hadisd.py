"""
Series of helper functions for reading and processing HADISD station data.
"""
import datetime
import gzip
import os
import shutil
import sys

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

import hccpml.vendor.hadisdhcalc.CalcHums as calc_hums
import hccpml.vendor.hadisdhcalc.CalcTw_Warren_NEWT as calc_Tw

# -----
# Globals
# -----
# The HADISD station data variable names
# Includes calculated variables for this project
HADISD_DATA_VARIABLES = [
    "temperatures",
    "dewpoints",
    "slp",
    "stnlp",
    "windspeeds",
    "winddirs",
    "wind_gust",
    "total_cloud_cover",
    "low_cloud_cover",
    "mid_cloud_cover",
    "high_cloud_cover",
    "precip1_depth",
    "precip2_depth",
    "precip3_depth",
    "precip6_depth",
    "precip9_depth",
    "precip12_depth",
    "precip15_depth",
    "precip18_depth",
    "precip24_depth",
    "cloud_base",
    "past_sigwx1",
    "wet_bulb_temperatures",
    "wet_bulb_anomalies",
    "wet_bulb_climatology"
]


# -----
# NetCDF file functions
# -----
def remove_reporting_stats(ds):
    """
    Reporting_stats is not currently used so remove.
    Reference: https://www.metoffice.gov.uk/hadobs/hadisd/hadisd_v340_2023f_product_user_guide.pdf
    """
    return ds.drop_vars('reporting_stats')


def add_flagged_obs(ds):
    """
    Values removed from each variables by the QC tests are stored in flagged_obs data variable.
    Restore the original obervations so they can be used as predictor variable.
    Reference: https://www.metoffice.gov.uk/hadobs/hadisd/hadisd_v340_2023f_product_user_guide.pdf
    """
    # attach a coordinate to flagged dimension so labels can be added
    # labels from user guide linked in docstring
    # TODO: no flags for cloud base, wind_gust or past sigwx1 - why?
    flagged_labels = [
        'temperatures', 'dewpoints', 'slp',
        'stnlp', 'windspeeds', 'winddirs', 
        'total_cloud_cover', 'low_cloud_cover', 'mid_cloud_cover', 
        'high_cloud_cover', 'precip1_depth', 'precip2_depth', 
        'precip3_depth', 'precip6_depth', 'precip9_depth',
        'precip12_depth', 'precip15_depth', 'precip18_depth', 
        'precip24_depth'
    ]
    ds = ds.assign_coords(flagged=("flagged", flagged_labels))
    
    # iterate through and add flagged data to each variable
    for var_name in ds['flagged'].data:
        flagged_var = ds['flagged_obs'].sel(flagged=var_name)
        qcd_var = ds[var_name]

        # fill nas in flagged variable with data from qc'd variable
        # then replace qc'd variable data
        ds[var_name].data = flagged_var.fillna(qcd_var).data

        # not all flagged values are available in flagged obs. Why?
        # TODO: understand why and replace with observations if possible
        ds[var_name][ds[var_name] == ds[var_name].attrs['flagged_value']] = np.nan

    return ds


def set_cloud_cover_missing_to_na(ds):
    """
    Cloud cover fields have missing data in flagged_obs set to -999. Cannot find this referenced in
    the data, so manually replace
    """
    varname_val_map = {
        'total_cloud_cover': -999., 
        'low_cloud_cover': -999., 
        'mid_cloud_cover': -999.,
        'high_cloud_cover': -999.
    }

    for var_name, miss_val in varname_val_map.items():
        ds[var_name][ds[var_name] == miss_val] = np.nan
    
    return ds


def add_station_id(ds):
    """Replace existing station id (uninterpretable) field with attribute (WMO-WBAN)."""
    del ds['station_id']
    ds = ds.expand_dims({'id': [ds.attrs['station_id']]})

    return ds


def reindex_time(ds, date_rng):
    """
    To reduce file size time points with missing data are removed from ISDH station data. This
    function restores the time index to an hourly time series for the given range. Missing data
    is stored as `np.NaN`.
    """
    # load if stored lazily, as this speeds up reindexing considerably
    # KW: just checking the data_range is working as expected as to me this is
    # creating a range only between the first two hours of the record.
    # Do you actually want to initialise the dates with hours explicitly to
    # begin with?
    # pd.date_range(dt.datetime(1970,1,1,0,0),dt.datetime(2023,12,31,23,0), freq='h')
    ds = ds.load()
    # KW I don't think you need the - datetime.timedelta bit as this
    # produces a range 1 day short
    #date_obj = pd.date_range(date_rng[0], date_rng[1] - datetime.timedelta(days=1), freq='h')
    date_obj = pd.date_range(date_rng[0], date_rng[1], freq='h')

    return ds.reindex({'time': date_obj})


def estimate_wet_bulb_temperature(airtemp, dewpoint):
    """Calculate the wet bulb temperature from the air temperature (T) and dewpoint (Td). Assumes
    standard pressure (P) of 1013 hPa. Returns the wet bulb temperature (Tw) in Kelvin."""
    Diffs = airtemp - dewpoint
    wbt = dewpoint.copy()
    wbt[Diffs != 0] = calc_Tw.wet_bulb_temperature(
        101300., 
        airtemp[Diffs != 0].data + 273.15,
        calc_hums.sh(dewpoint[Diffs != 0],
            airtemp[Diffs != 0],1013.,roundit=False).data / 1000.,
            saturation='adiabatic',phase='liquid',pseudo_method='polynomial') - 273.15 
    
    return wbt


def remask_array(tmparray, mdi):
    """
    Function to create masked arrays or if object is already a masked array then ensure 
    missing values are filled with desired mdi and reset mask to ISDMDI.
    Functions like ma.mean return a filled value of 0 rather than set fill_value!
    """
    if (isinstance(tmparray, ma.MaskedArray)):
        tmparray = tmparray.filled(mdi)

    tmparray = ma.masked_values(tmparray, mdi)
    tmparray.fill_value = mdi
    return tmparray


def make_climatologies(hourlies, smoother, mdi, climatology_type="smooth"):
    """
    This function calculates hourly climatologies over the period.
    
    It produces a 1 year simple mean climatology for each hour and could (but doesn't)
    return it. It produces a smoothed hour climatology over an n day length window for each hour.
    It could (but doesn't) return this single year of climatological hours. It produces a repeated
    climatology filling the entire data range and returns.

    Notes
    -----
    There is no minimum data presence criteria for an hour climatolgy to be calculated. This could be considered?
    
    Parameters
    ----------
    hourlies : `np.ma` array
        A masked array of observations of size n years x 366 days by 24 hours. Missing data is
        identified by `ISDMDI` and marked as `True` in the mask.
    smoother : int
        Length in days for the smoothing window which is centred on candidate day, wrapping across
        December to January as necessary.
    ISDMDI : float
        Missing data indicator. Typically -1e30.
            
    Returns
    -------
    single_climatology : `np.ma`
        A masked array of size 366 (days) by 24 (hours) masked array of floats of the climatology field. There should
        not be any missing data!
    climatology : `np.ma`
        A masked array of n (years) by 366 (days) by 24 (hours) array of repeated hour smoothed climatologies. This is
        returned for convenience as the single_climatology is repeated for each year. Each year has a 29th Feb, which
        may need to be removed for future use.
    """
    
    # Get number of years from hourlies 1st [0] dimensions
    n_years = np.shape(hourlies)[0]

    # Check that the climatology type is valid argument.
    if climatology_type not in ("smooth", "spiky"):
        raise ValueError(f"Climatology type {climatology_type}. Valid options are 'smooth' or 'spiky'")
    
    # Set smoothing paramters
    low_half = int(np.floor(smoother/2.))

    # Create climatological mean for each hour of each day = 366 days 
    # rows by 24 hours columns
    # Using entire period here (avoids any inbase/outofbase issues) - 
    # could use 1991-2020
    # Should really apply some kind of minimum data completeness threshold 
    # but this would then 'remove' data
    spiky_climatology = ma.mean(hourlies, axis=0)
    # Ensure masked array is as expected after insert (fill_value can change)
    spiky_climatology = remask_array(spiky_climatology, mdi)

    # Create a smoother climatolology by making each hour the mean of 
    # the +/- 2 day-hours either side
    # e.g. Jan 10th 17:00 is mean of Jan 8th, 9th, 10th, 11th, 12th 17:00 hours
    smooth_climatology = np.zeros((366, 24), dtype=float)
    smooth_climatology[:,:] = mdi
    smooth_climatology = remask_array(smooth_climatology, mdi)
    
    for dd in np.arange(366):
        #print(dd)
        if (dd > low_half-1) & (dd < (366 - low_half)):
            smooth_climatology[dd,:] = ma.mean(
                spiky_climatology[dd-low_half:dd+(low_half+1),:], axis=0)
        
        elif (dd <= low_half-1):
            smooth_climatology[dd,:] = ma.mean(
            ma.reshape(ma.append(spiky_climatology[365-((low_half-1)-dd):,:], 
                                spiky_climatology[0:dd+(low_half+1),:]),
                                (smoother, 24)), axis=0)
        elif (dd >= (366 - low_half)):
            smooth_climatology[dd,:] = ma.mean(
            ma.reshape(ma.append(spiky_climatology[dd-low_half:,:], 
                                spiky_climatology[0:(low_half-(365-dd)),:]), 
                                (smoother, 24)), axis=0)

    # Ensure masked array is as expected after insert (fill_value can change)
    smooth_climatology = remask_array(smooth_climatology, mdi)
    
    if (climatology_type == 'spiky'):
        single_climatology = spiky_climatology
    elif (climatology_type == 'smooth'):
        single_climatology = smooth_climatology

    # Create full timeseries of repeated climatologies but still including a
    # Feb 29th every year
    # Going from a 366 by 24 hour array to an NYear by 366 day by 24 hour array
    climatology = np.tile(single_climatology, (n_years, 1, 1))
    climatology = remask_array(climatology, mdi)

    return single_climatology, climatology


def make_anomalies_climatology(xhourlies, smoother):
    """
    This function calculates hourly anomalies and climatologies over the 
    entire record using the `make_climatologies()` function.
    
    It returns a repeated climatology filling the entire data range as 
    well as anomalies where the climatological hour has been subtracted.

    Parameters
    ----------
    xhourlies : xarray.DataArray
        The hourly oberservation data encapsulated in a DataArray. The data should have a
        shape (N, ) (i.e. it is a 1-D array), where N is the total number of hourly observations in
        a given time period. The DataArray should have a dimension and index called time of length N.
        Time should be an datetime index of the hourly time points at which the observations were taken.
        It is expected that the time index and data is complete for the period (i.e. no missing hours)
        and missing data is identified by np.NaN.
    smoother : int
        Size of the smoothing window in days. The smoothing window is centred on the candidate day,
        wrapping across December to January as necessary.
            
    Returns
    -------
    xanomalies : xarray.DataArray
        The anomalies of the observations (i.e. actual - climatology). It has the same dimensions
        and index as xhourlies. Missing data is identified with `np.nan`.
    xclimatology : xarray.DataArray
        The hourly climatology for repeated for each year in the time period.  It has the same dimensions
        and index as xhourlies. Missing data is identified with `np.nan`.
    """
    # Missing value for masked arrays
    ISDMDI = -1e30
    
    # Get date parameters
    # daylist is the unique calendar days in the time index of xhourlies  
    # Note, we expect xhourlies time index to be a pandas.DateTimeIndex
    # daylist is a pandas.PeriodIndex  
    daylist = xhourlies.indexes["time"].to_period("d").drop_duplicates()  
    # Want the total number of days in the index, i.e. the len of the daylist  
    n_days = len(daylist)  
    # Number of **full** years in time period  
    n_years = (daylist.max().year - daylist.min().year) + 1
    # Number of hours taken from the time index
    n_hours = len(xhourlies["time"]) 

    # create numpy masked array from the xarray data object with NaNs
    hourlies = np.copy(xhourlies.values)
    hourlies[np.isnan(hourlies)] = ISDMDI
    hourlies = remask_array(hourlies, ISDMDI)

    # Reshape timeseries to NDays rows by 24 hours columns (including 29th Febs)
    hourlies = ma.reshape(hourlies, (n_days, 24))
    
    # Locate all non-leap year 28th Febs and pop an extra day of 24 MDIs in 
    # These are pseudo 29th Febs.
    nonleap = '-02-28'
    # an array of location (in days) where there are Feb 28ths, (days since 
    # 1st Jan 1973)
    feb28ths = [ss for ss, s in enumerate(daylist) if nonleap in str(s)]
    # adjusts for actual location of 28th Febs in days vs array where we have 
    # added fake 29th Febs
    popper = 0 
    
    for dd in feb28ths:
        # check its there isn't already a Feb 29th
        if ('-02-29' not in str(daylist[dd+1])):
            hourlies = np.insert(hourlies, dd+1+popper, 
                                    np.tile(ISDMDI, 24), axis=0)
        
            # Readjust PopAdjuster for next Feb 28th to account for added 
            # row for pseudo-29th Feb
            popper += 1
            
    # Reshape timeseries to NYrs blocks, by 366 days rows and 24 hours columns
    hourlies = ma.reshape(hourlies, (n_years, 366, 24))
    # Ensure masked array is as expected after insert (fill_value can change)
    hourlies = remask_array(hourlies, ISDMDI)
    
    single_climatology, climatology = make_climatologies(hourlies, smoother, 
                                                         ISDMDI)

    # Where non-missing, subtract climatological hour mean for each hour of 
    # each day. Creates 366 days rows, by 24 hours columns, by NYrs (blocks) 
    # - masked data array
    anomalies = hourlies - single_climatology
    # Ensure masked array is as expected after insert (fill_value can change)
    anomalies = remask_array(anomalies, ISDMDI)
    
    # Reshape back to NDays rows by 24 hours columns
    anomalies = ma.reshape(anomalies, (n_days+popper, 24))
    climatology = ma.reshape(climatology, (n_days+popper, 24))
    
    # Locate pseudo-29th Feb for each non-leap year and remove them - use 
    # feb28ths array from above
    # No PopAdjuster necessary as we're going back to actual length of days
    for dd in feb28ths:
        # check it  isn't a real Feb 29th
        if ('-02-29' not in str(daylist[dd+1])):
            anomalies = np.delete(anomalies, dd+1, axis=0)
            climatology = np.delete(climatology, dd+1, axis=0)
                    
    # Reshape back to nHours complete masked array.
    anomalies = ma.reshape(anomalies, n_hours)
    single_climatology = ma.reshape(single_climatology, 366*24)
    climatology = ma.reshape(climatology, n_hours)
    
    # Ensure masked array is as expected after insert (fill_value can change)
    anomalies = remask_array(anomalies, ISDMDI)
    climatology = remask_array(climatology, ISDMDI)

    # Create xarray data objects of anomalies and climatology to return
    xanomalies = xhourlies.copy()
    xclimatology = xhourlies.copy()
    # Need to be sure flags aren't going wrong here
    # For a masked array, access the original values using ma.data. The array can be filled using
    # ma.filled(). I think this is what we want to do here setting the fill value to np.nan
    # Then we don't have to change any values in xanomalies or xclimatology.
    # The current approach will work but is perhaps not the optimal way of using masked arrays.
    xanomalies.values = anomalies.data
    xanomalies[np.isclose(xanomalies, ISDMDI)] = np.nan
    xclimatology.values = climatology.data
    xclimatology[np.isclose(xclimatology, ISDMDI)] = np.nan
    
    return xanomalies, xclimatology


# KW I think you need to include the hours in here (at least for reindex to work)
def read_station(fp, date_range=(datetime.datetime(1970,1,1,0,0), datetime.datetime(2023,12,31,23,0)), smoother=31):
    """Read a single HADISD station data set and apply transformations. This function should be used
    to read all station data if needing to perform machine learning. The function applies the
    following transformations in order:
    - Remove reporting statistics: removes the (unused) reporting statistics from the station data.
        See `remove_reporting_stats` for details.
    - Restore flagged observations: restores QC flagged observations into the original time series.
        See `add_flagged_obs` for details.
    - Fix missing cloud cover observations: sets the missing values in the cloud cover variables to
        `np.NaN`. See `set_cloud_cover_missing_to_na` for details.
    - Make the time series hourly: re-indexes the time series for all obversation variables to a
        complete hourly time series for the given date range. See `reindex_time` for details.
    - Calculate the wet bulb temperature: estimate the wet bulb temperature from the air temperature
        and dewpoints. The new variable is called `wet_bulb_temperatures`. See
        `estimate_wet_bulb_temperature` for details.
    - Calculate wetbulb climatology and anomalies: the climatology and anomalies are calculated for
        the given date range. The new variables are called `wet_bulb_anomalies` and
        `wet_bulb_climatology`. See `make_anomalies_climatology` for details.
    - Remove redundant dimensions: squeeze the data set to remove any redundant dimensions from the
        data set.

    It is important to note that this function is used downstream to read the HADISD station data
    in the `hadisd-data-conversion` workflow. Please modify with care!

    Parameters
    ----------
    fp : str
        The filepath to the original HADISD station data. Can be a NetCDF file or a gzipped NetCDF
        file.
    date_range : tuple
        A tuple of two datetime objects. The first is the start date and the second is the end date
        of the date range. The time series will be re-indexed to this date range and data outside
        of this range will be dropped. The default is 00:00 on 1st January 1970 to 23:00 on 31st
        December 2023.
    smoother : int
        The smoothing windows in days for calculating the climatology and anomalies (of the wet
        bulb temperature). The default is 31.

    References
    ----------
    https://www.metoffice.gov.uk/hadobs/hadisd/hadisd_v340_2023f_product_user_guide.pdf
    """

    # We make life hard sometimes. Is this actually a .nc.gz file?
    # Have to load the dataset into memory so that we can operate on it
    if fp.endswith('.gz'):
        # Unzip the file
        with gzip.open(fp) as fh:
            isdh_ds = xr.load_dataset(fh)
    else:
        isdh_ds = xr.load_dataset(fp)

    isdh_ds = remove_reporting_stats(isdh_ds)
    isdh_ds = add_flagged_obs(isdh_ds)
    isdh_ds = set_cloud_cover_missing_to_na(isdh_ds)
    isdh_ds = reindex_time(isdh_ds, date_range)
    # Estimate wet bulb temperatures
    isdh_ds['wet_bulb_temperatures'] = estimate_wet_bulb_temperature(
        airtemp=isdh_ds['temperatures'], 
        dewpoint=isdh_ds['dewpoints']
        )
    
    # Create anomaly and climatology arrays for wet-bulb
    # This requires a datetime list of days (not hours) for complete record
    isdh_ds['wet_bulb_anomalies'], isdh_ds['wet_bulb_climatology'] = \
        make_anomalies_climatology(isdh_ds['wet_bulb_temperatures'], smoother)

    # Remove redundant dimensions
    isdh_ds = isdh_ds.squeeze()
    # FIXME: removing this for now as it re-dimensions the data.
    # add station id back in
    # isdh_ds = add_station_id(isdh_ds)

    return isdh_ds


# TODO: remove this function when confirmed it is no longer used.
def read_isdh(station_ids, data_loc=os.path.join(os.environ.get("SCRATCH"), 'isdh')):
    """
    reads isdh data 
    """
    # where isdh source data is stored
    isdh_source_loc = '/data/users/hadkw/WORKING_HADISDH_9120/UPDATE2023/HADISDTMP'

    if type(station_ids) is str:
        Warning('station_id is a string. Converting to list.')
        station_ids = [station_ids]

    # data are stored as zipped files on Kate's datadir. First check if they are available in data_loc. If not, 
    # unzip then write to data_loc
    isdh_name_format = 'hadisd.3.4.0.2023f_19310101-20240101_{station_id}.nc'
    isdh_fns = [isdh_name_format.format(station_id=station_id_i) for station_id_i in station_ids]
    isdh_fps = [os.path.join(data_loc, fn) for fn in isdh_fns]

    isdh_fps_to_unzip = [fp for fp in isdh_fps if not os.path.isfile(fp)]

    # decompress original gz file
    os.makedirs(data_loc, exist_ok=True)
    for fp in isdh_fps_to_unzip:
        print(f'Writing {fp}...')
        fn = os.path.basename(fp)
        source_fp = os.path.join(isdh_source_loc, fn) + '.gz'

        with gzip.open(source_fp, 'rb') as f_in:
            with open(fp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # read
    isdh_ds_ls = [read_station(fp) for fp in isdh_fps]

    return xr.concat(isdh_ds_ls, dim='id')


# -----
# Metadata readers and processes
# -----
def read_station_metadata_filtered(metadata_fp):
    """
    Reads station metadata in the "filtered" format. This metadata file has been generated by Kate
    Willet's scripts and filters the stations by a number of criteria. Similar to the source files,
    the metadata file is fixed width with the following columns:
        - 'station_id' (type=str, width=11): the station id, a concatenation of the WMO and WBAN numbers.
            Must be a string because numbers start with leading 0's.
        - 'latidude' (type=float, width=9): the latitude of the station.
        - 'longitude' (type=float, width=10): the longitude of the station.
        - 'elevation' (type=float, width=7): the elevation of the station.
        - 'country_code' (type=str, width=3): the country code of the station.
        - 'station_name' (type=str, width=30): the station name.

    Note that column headers are not specified in this file. On read, the station_id is split into
    WMO and WBAN parts and the station_id is recreated in the format [WMO]-[WBAN].
    
    This function is based on the original function used by Kate Willet. It uses panads to read the
    metadata.


    Inputs 
    ------
    station_fp: path-like
        Path to the metadata file.
    
    Outputs
    -------
    pandas.DataFrame
        DataFrame containing the metadata in fixed-width format.
    """

    # Fixed width format with no header (!)
    # Need widths, names, and types
    # The station_id is split into WMO and WBAN
    col_widths = [6, 5, 9, 10, 7, 3, 30]
    col_names = ["WMO", "WBAN", "latitude", "longitude", "elevation", "country_code", "station_name"]
    col_types = {
        "WMO": str,
        "WBAN": str,
        "latitude": float,
        "longitude": float,
        "elevation": float,
        "country_code": str,
        "station_name": str
    }
    metadata = pd.read_fwf(metadata_fp, widths=col_widths, names=col_names, dtype=col_types)
    # Add the station_id
    metadata["station_id"] = metadata["WMO"].str.cat(metadata["WBAN"], sep="-")

    return metadata

def assign_station_gridbox(lat, lon, box_width=5.):
    """
    Given a lat and lon (or array of) return:
        - a df of gridbox numbers and centres for each station
    
    The df columns are:
        - gridbox_id: numbers from 0 to n starting at 180W and 90N tracking along 
            the corridor and down the stairs
        - gridbox_<>_centres lon (-180 to 180), lat -90 to 90.

    Gridbox resolution is set here by box_width to 5 by 5 degrees, but user can set as desired.
    This has been set up and tested only when lat and lon widths are the same.
    The southern/western boundary is included in the gridbox e.g., A station 
    at -180.00W, 89.0N sits in gridbox 0 and gridbox centre 177.5W, 87.5N. A
    station at 179.0E and -85.0S sits in gridbox 2591 and gridbox centre 
    177.5W, -82.5S.

    Inputs 
    ------
    lat :: float or np.array of floats
        latitude in decimal form
    lon :: float or np.array of floats
        longitude in decimal form
    box_width :: float (default = 5.)
        Width of gridbox in whole degrees of latitude and longitude
    
    Outputs
    -------
    gridbox_info :: pandas.DataFrame
        DataFrame containing the gridbox_id ints, gridbox_lat_centres floats and gridbox_lon_centres floats
    """
    # settable parameters defining gridbox size - theoretically can work with other values, 1, 10 etc.
    lat_width = box_width
    lon_width = box_width
    
    # work out gridbox parameters based on the above lat_width and lon_width
    num_lats = int(180 / lat_width)
    num_lons = int(360 / lon_width)

    gridbox_info = pd.DataFrame()

    # Create identification array of gridbox numbers from 0 (northwest corner) to 2591 
    # (southeast corner) tracking along the corridor and down the stairs
    gridbox_numbers = np.reshape(np.arange(num_lats*num_lons), (num_lats, num_lons))

    # Create identification array of gridbox latitude centres from -87.5 (south) to  
    # 87.5 (north), repeating for every longitude.
    gridbox_lat_centres = np.flipud(np.transpose(np.tile((
        (np.arange(num_lats)*lat_width)-(90. - (lat_width/2.))), (num_lons, 1))))
    
    # Create identification array of gridbox longitude centres from -177.5 (west) to  
    # 177.5 (east), repeating for every latitude.
    gridbox_lon_centres = np.tile((
        (np.arange(num_lons)*lon_width)-(180. - (lon_width/2.))), (num_lats, 1))
    
    # Map the latitudes onto row indices from 0 (87.5N) to 35 (-87.5S)
    # Add a catch for anything at 90N as this gets a -1 box
    map_lats = np.array([(-(np.floor((np.atleast_1d(lat) + 90.) / lat_width).astype(int)) + (num_lats-1))])
    map_lats[map_lats < 0] = 0

    # Map the longitudes onto column indices from 0 (-177.5W) to 71 (177.5E)
    # Add a catch for anything at 180E as this gets a 72 box but should be 0
    map_lons = np.array([(np.floor((np.atleast_1d(lon) + 180.) / lon_width).astype(int))])
    map_lons[map_lons > (num_lons-1)] = 0 

    # Assign gridbox number, lat centre and lon centre to station
    gridbox_info["gridbox_id"] = gridbox_numbers[tuple(map_lats), tuple(map_lons)][0]
    gridbox_info["gridbox_lat_centres"] = gridbox_lat_centres[tuple(map_lats), tuple(map_lons)][0]
    gridbox_info["gridbox_lon_centres"] = gridbox_lon_centres[tuple(map_lats), tuple(map_lons)][0]

    # How should this information be returned
    return gridbox_info      

