#--------------------------------
# Functions
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import spatial

#--------------------------------
# utility
#--------------------------------

def get_index_latlon(
        loni,
        lati,
        lon_arr,
        lat_arr,
        search_range=5.0,
        ):
    """Get the index of the location (loni, lati) in an array of
       locations (lon_arr, lat_arr)

    :loni:          (float) Longitude of target location
    :lati:          (float) Latitude of target location
    :lon_arr:       (numpy array) array of Longitude
    :lat_arr:       (numpy array) array of Latitude
    :search_range:  (float) range of longitude and latitude in degrees for faster search

    """
    lon_mask = (lon_arr>=loni-search_range) & (lon_arr<=loni+search_range)
    lat_mask = (lat_arr>=lati-search_range) & (lat_arr<=lati+search_range)
    lonlat_mask = lon_mask & lat_mask
    lon_sub = lon_arr[lonlat_mask]
    lat_sub = lat_arr[lonlat_mask]
    pts = np.array([loni,lati])
    tree = spatial.KDTree(list(zip(lon_sub, lat_sub)))
    p = tree.query(pts)
    cidx = p[1]
    idx = np.argwhere(lon_arr==lon_sub[cidx])
    for i in idx[0][:]:
        if lat_arr[i] == lat_sub[cidx]:
            out = i
            break
    return out

#--------------------------------
# Great circle
#--------------------------------

def gc_radius():
    """Return the radius of Earth in km

    :returns:   (float) radius of Earth in km

    """
    return 6371.0

def gc_angle(
        lon0,
        lat0,
        lon1,
        lat1,
        ):
    """Calculate the angle counterclockwise from east.

    :lon0:      (float) longitude of point 1 in degrees
    :lat0:      (float) latitude of point 1 in degrees
    :lon1:      (float) longitude of point 2 in degrees
    :lat1:      (float) latitude of point 2 in degrees
    :returns:   (float) angle in degrees

    """
    dlon_r = np.radians(lon1-lon0)
    dlat_r = np.radians(lat1-lat0)
    angle = np.arctan2(dlat_r, dlon_r)
    return angle

def gc_angles(
        lon,
        lat,
        ):
    """A wrapper of gc_angle to compute the angle counterclockwise from east for an array of lon and lat

    :lon:   (numpy array) array of longitudes
    :lat:   (numpy array) array of latitudes

    """
    lat0 = np.zeros(lat.size)
    lon0 = np.zeros(lon.size)
    lat1 = np.zeros(lat.size)
    lon1 = np.zeros(lon.size)
    lat0[1:-1] = lat[0:-2]
    lat1[1:-1] = lat[2:]
    lon0[1:-1] = lon[0:-2]
    lon1[1:-1] = lon[2:]
    angles = gc_angle(lon0, lat0, lon1, lat1)
    angles[0] = angles[1]
    angles[-1] = angles[-2]
    return angles

def gc_distance(
        lon0,
        lat0,
        lon1,
        lat1,
        ):
    """Calculate the great circle distance (km) between two points [lon0, lat0] and [lon1, lat1]
    http://www.movable-type.co.uk/scripts/latlong.html

    :lon0:      (float) longitude of point 1 in degrees
    :lat0:      (float) longitude of point 1 in degrees
    :lon1:      (float) longitude of point 2 in degrees
    :lat1:      (float) longitude of point 2 in degrees
    :returns:   (numpy array) longitude and latitude

    """
    radius = gc_radius() # km
    dlat_r = np.radians(lat1 - lat0)
    dlon_r = np.radians(lon1 - lon0)
    lat0_r = np.radians(lat0)
    lat1_r = np.radians(lat1)
    a = (np.sin(dlat_r / 2) * np.sin(dlat_r / 2) +
         np.cos(lat0_r) * np.cos(lat1_r) *
         np.sin(dlon_r / 2) * np.sin(dlon_r / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d

def gc_interpolate(
        lon0,
        lat0,
        lon1,
        lat1,
        npoints,
        ):
    """Interpolate on a great circle between two points [lon0, lat0] and [lon1, lat1]
    http://www.movable-type.co.uk/scripts/latlong.html

    :lon0:      (float) longitude of point 1 in degrees
    :lat0:      (float) longitude of point 1 in degrees
    :lon1:      (float) longitude of point 2 in degrees
    :lat1:      (float) longitude of point 2 in degrees
    :npoints:   (int) number of points for interpolation
    :returns:   (numpy array) longitude and latitude

    """
    radius = gc_radius() # km
    frac = np.linspace(0, 1, npoints)
    lon0_r = np.radians(lon0)
    lat0_r = np.radians(lat0)
    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    delta = gc_distance(lon0, lat0, lon1, lat1) / radius
    a = np.sin((1 - frac) * delta) / np.sin(delta)
    b = np.sin(frac * delta) / np.sin(delta)
    x = a * np.cos(lat0_r) * np.cos(lon0_r) + b * np.cos(lat1_r) * np.cos(lon1_r)
    y = a * np.cos(lat0_r) * np.sin(lon0_r) + b * np.cos(lat1_r) * np.sin(lon1_r)
    z = a * np.sin(lat0_r) + b * np.sin(lat1_r)
    lat_out = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon_out = np.arctan2(y, x)
    return np.degrees(lon_out), np.degrees(lat_out)

#--------------------------------
# plot
#--------------------------------

def plot_basemap(
        region='Global',
        axis=None,
        ):
    """Plot basemap

    :region:    (str) region name
    :axis:      (matplotlib.axes, optional) axis to plot figure on
    :return:    (Basemap) basemap

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # plot basemap
    switcher = {
            'Global': _plot_basemap_global,
            'Arctic': _plot_basemap_arctic,
            'LabSea': _plot_basemap_labsea,
            'TropicalPacific': _plot_basemap_tropicalpacific,
            'TropicalAtlantic': _plot_basemap_tropicalatlantic,
            }
    if region in switcher.keys():
        return switcher.get(region)(axis)
    else:
        raise ValueError('Region \'{:s}\' not found.\n'.format(region) \
                + '- Supported region names:\n' \
                + '  ' + ', '.join(switcher.keys()))

def _plot_basemap_global(axis):
    """Plot basemap for global region

    """
    # global map
    m = Basemap(projection='cyl', llcrnrlat=-80., urcrnrlat=80.,
            llcrnrlon=20., urcrnrlon=380., ax=axis)
    # plot coastlines, draw label meridians and parallels.
    m.drawcoastlines(zorder=3)
    m.drawmapboundary(fill_color='lightgray')
    m.fillcontinents(color='gray',lake_color='lightgray', zorder=2)
    m.drawparallels(np.arange(-80.,81.,30.), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-180.,181.,60.), labels=[1,0,0,1])
    return m

def _plot_basemap_arctic(axis):
    """Plot basemap for Arctic

    """
    m = Basemap(projection='npstere', boundinglat=50, lon_0=0,
                resolution='l', ax=axis)
    # plot coastlines, draw label meridians and parallels.
    m.drawcoastlines(zorder=3)
    m.drawmapboundary(fill_color='lightgray')
    m.fillcontinents(color='gray',lake_color='lightgray', zorder=2)
    m.drawparallels(np.arange(-80.,81.,10.), labels=[0,0,0,0])
    m.drawmeridians(np.arange(-180.,181.,20.), labels=[1,0,1,1])
    return m

def _plot_basemap_region(
        axis,
        lon_ll,
        lat_ll,
        lon_ur,
        lat_ur,
        projection,
        ):
    """Plot basemap for region

    :axis:          (matplotlib.axes, optional) axis to plot figure on
    :lon_ll:        (float) longitude at lower-left in degrees
    :lat_ll:        (float) latitude at lower-left in degrees
    :lon_ur:        (float) longitude at upper-right in degrees
    :lat_ur:        (float) latitude at upper-right in degrees
    :projection:    (str) projection type
    :return:        (Basemap) basemap

    """
    # regional map
    lon_c = 0.5*(lon_ll+lon_ur)
    lat_c = 0.5*(lat_ll+lat_ur)
    m = Basemap(projection=projection, llcrnrlon=lon_ll, llcrnrlat=lat_ll,
                urcrnrlon=lon_ur, urcrnrlat=lat_ur, resolution='l',
                lon_0=lon_c, lat_0=lat_c, ax=axis)
    # plot coastlines, draw label meridians and parallels.
    m.drawcoastlines(zorder=3)
    m.drawmapboundary(fill_color='lightgray')
    m.fillcontinents(color='gray',lake_color='lightgray', zorder=2)
    m.drawparallels(np.arange(-80.,81.,10.), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-180.,181.,10.), labels=[1,0,0,1])
    return m

def _plot_basemap_labsea(axis):
    """Plot basemap for Labrador Sea

    """
    return _plot_basemap_region(axis, lon_ll=296.0, lat_ll=36.0, \
                                lon_ur=356.0, lat_ur=70.0, projection='cass')

def _plot_basemap_tropicalpacific(axis):
    """Plot basemap for Tropical Pacific

    """
    return _plot_basemap_region(axis, lon_ll=130.0, lat_ll=-20.0, \
                                lon_ur=290.0, lat_ur=20.0, projection='cyl')

def _plot_basemap_tropicalatlantic(axis):
    """Plot basemap for Tropical Atlantic

    """
    return _plot_basemap_region(axis, lon_ll=320.0, lat_ll=-20.0, \
                                lon_ur=370.0, lat_ur=20.0, projection='cyl')


