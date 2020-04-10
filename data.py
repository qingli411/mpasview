#--------------------------------
# MPAS data type
#--------------------------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .functions import plot_basemap
from .dtype import UArray2D, UMesh

#--------------------------------
# MPAS
#--------------------------------

class MPASOMap(UArray2D):
    """A data type describing a map of MPAS-Ocean field

    """

    def __init__(
            self,
            data = np.nan,
            name = '',
            units = '',
            lon = np.nan,
            lat = np.nan,
            mesh = None,
            ):
        """Initialization of MPASOMap

        :data:      (list or numpy array) data array
        :name:      (str) name of data
        :units:     (str) units of data
        :lon:       (list or numpy array) longitude array in degrees
        :lat:       (list or numpy array) latitude array in degrees
        :mesh:      (UMesh) mesh object

        """
        self.mesh = mesh
        if mesh is None:
            assert lon is not None, 'Either pass in (lon, lat) pairs or the mesh.'
            assert lat is not None, 'Either pass in (lon, lat) pairs or the mesh.'
            super(MPASOMap, self).__init__(
                    data=data, name=name, units=units,
                    x=lon, xname='lon', xunits='degree_east',
                    y=lat, yname='lat', yunits='degree_north')
        else:
            #  TODO: Support mesh info <20200409, Qing Li> #
            pass

    def __repr__(self):
        """Formatted print

        """
        if self.mesh is None:
            return super(MPASOMap, self).__repr__() + '\n  mesh: none'
        else:
            return super(MPASOMap, self).__repr__() + '\n  mesh: ' + self.mesh.name


    def plot(
            self,
            axis = None,
            region = 'Global',
            levels = None,
            cmap = 'viridis',
            colorbar = True,
            **kwargs,
            ):
        """Plot figure

        :axis:      (matplotlib.axes, optional) axis to plot figure on
        :region:    (str) region name
        :leveles:   (list, optional) list of levels
        :cmap:      (str, optional) colormap
        :colorbar:  (bool) do not add colorbar if False
        :**kwargs:  (keyword arguments) other arguments
        :return:    (Basemap) figure handle

        """
        # use curret axis if not specified
        if axis is None:
            axis = plt.gca()
        # print message
        print('Plotting \'{:s}\' map in the \'{:s}\'...'.format(self.data.name+' ('+self.data.units+')', region))
        # basemap
        m = plot_basemap(region=region, axis=axis)
        # mask out nan
        nan_mask = (~ np.isnan(self.data))
        # preprocess data
        data = self.data[nan_mask]
        lat = self.y[nan_mask]
        lon = self.x[nan_mask]
        # manually mapping levels to the colormap if levels is passed in,
        if levels is not None:
            bounds = np.array(levels)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        else:
            norm = None
        # simple plot if mesh is not defined
        if self.mesh is None:
            x, y = m(lon, lat)
            fig = m.contourf(x, y, data, tri=True, levels=levels, extend='both',
                        norm=norm, cmap=plt.cm.get_cmap(cmap), **kwargs)
        #  TODO: support pcolor with mesh info <20200409, Qing Li> #
        # add colorbar
        if colorbar:
            cb = m.colorbar(fig, ax=axis)
            cb.set_label('{} ({})'.format(self.data.name, self.data.units))
            cb.formatter.set_powerlimits((-4, 4))
            cb.update_ticks()
        return m

#--------------------------------
# MPASOMesh
#--------------------------------

class MPASOMesh(UMesh):

    """A data type for MPAS-Ocean mesh"""

    def __init__(self, filepath):
        """Initialization of MPASOMesh

        :filepath:  (str) path of the MPAS-Ocean mesh file

        """
        self._filename = filename
        # load file
        fmesh = xr.open_dataset(self._filename)

        self.name = name
        # cells
        self.ncells = ncells
        self.cellid = cellid
        self.xcell = xcell
        self.ycell = ycell
        self.acell = area_cell
        self.nedges_cell = nedges_cell
        self.edges_cell = edges_cell
        # edges
        self.nedges = nedges
        self.edgeid = edgeid
        self.xedge = xedge
        self.yedge = yedge
        self.dc_edge = dc_edge
        self.dv_edge = dv_edge
        self.cells_edge = cells_edge
        # vertices
        self.nvertices = nvertices
        self.vertexid = vertexid
        self.xvertex = xvertex
        self.yvertex = yvertex
        self.adual = adual
        self.cells_vertex = cells_vertex
        self.edges_vertex = edges_vertex


