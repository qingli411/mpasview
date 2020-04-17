#--------------------------------
# plot
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon

#--------------------------------
# basemap
#--------------------------------

def plot_basemap(
        region='Global',
        axis=None,
        ):
    """Plot basemap

    :region:    (str) region name
    :axis:      (matplotlib.axes, optional) axis to plot the figure on
    :return:    (mpl_toolkits.basemap.Basemap)

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
    """Plot basemap for a particular region

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :lon_ll:        (float) longitude at lower-left in degrees
    :lat_ll:        (float) latitude at lower-left in degrees
    :lon_ur:        (float) longitude at upper-right in degrees
    :lat_ur:        (float) latitude at upper-right in degrees
    :projection:    (str) projection type
    :return:        (mpl_toolkits.basemap.Basemap)

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
    return _plot_basemap_region(axis, lon_ll=310.0, lat_ll=-20.0, \
                                lon_ur=380.0, lat_ur=20.0, projection='cyl')


#--------------------------------
# pseudocolor (pcolor) plot on unstructured grid
#--------------------------------

def ug_pcolor_cell(
        axis = None,
        data = np.nan,
        vertexid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        nedges_cell = np.nan,
        vertices_cell = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (cells)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :vertexid:      (array-like) vertex ID
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :nedges_cell:   (array-like) number of edges on cells
    :vertices_cell: (array-like) vertices on cells
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # patches
    patches = []
    ncell = nedges_cell.size
    for i in np.arange(ncell):
        vid = vertices_cell[i,:nedges_cell[i]]
        vidx = vid-1
        # TODO: assuming the vertex id is the index + 1 for now
        # which is much faster than the following code using vertexid
        # vidx = np.zeros(vid.size, np.int)
        # for j in np.arange(vid.size):
        #     vidx[j] = np.argwhere(vertexid==vid[j])
        xp = xvertex[vidx]
        yp = yvertex[vidx]
        patches.append(Polygon(list(zip(xp,yp))))
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    pc.set_array(data)
    fig = axis.add_collection(pc)
    return fig

def ug_pcolor_vertex(
        axis = None,
        data = np.nan,
        cellid = np.nan,
        xcell = np.nan,
        ycell = np.nan,
        cells_vertex = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (dual cells centered on vertices)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :cellid:        (array-like) cell ID
    :xcell:         (array-like) x-coordinate of cells
    :ycell:         (array-like) y-coordinate of cells
    :cells_vertex:  (array-like) cells on vertices
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # patches
    patches = []
    idx_mask = []
    nvertex = cells_vertex.shape[0]
    for i in np.arange(nvertex):
        cid = cells_vertex[i,:]
        cidx = cid-1
        # TODO: assuming the cell id is the index + 1 for now
        # which is much faster than the following code using cellid
        # cidx = np.zeros(cid.size, np.int)
        # for j in np.arange(cid.size):
        #     cidx[j] = np.argwhere(cellid==cid[j])
        if any(cidx == -1):
            idx_mask.append(i)
            continue
        xp = xcell[cidx]
        yp = ycell[cidx]
        patches.append(Polygon(list(zip(xp,yp))))
    data = np.delete(data, idx_mask)
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    pc.set_array(data)
    fig = axis.add_collection(pc)
    return fig
