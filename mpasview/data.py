#--------------------------------
# MPAS data type
#--------------------------------

import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .plot import plot_basemap, ug_pcolor_cell, ug_pcolor_vertex
from .utils import get_region_llrange

#--------------------------------
# MPASMesh
#--------------------------------

class MPASMesh:

    """A data type for MPAS-Ocean mesh"""

    def __init__(
            self,
            name = '',
            filepath = '',
            ):
        """Initialization of MPASOMesh

        :name:      (str) name of the mesh
        :filepath:  (str) path of the MPAS-Ocean mesh file

        """
        self.name = name
        self.filepath = filepath

        with xr.open_dataset(self.filepath) as fmesh:
            if fmesh.attrs['on_a_sphere'] == 'YES':
                self.on_sphere = True
            else:
                self.on_sphere = False
            self.maxedges_cell = fmesh.dims['maxEdges']
            self.vertexdegree  = fmesh.dims['vertexDegree']
            self.ncells        = fmesh.dims['nCells']
            self.nedges        = fmesh.dims['nEdges']
            self.nvertices     = fmesh.dims['nVertices']
            self.cellid        = fmesh.variables['indexToCellID']
            self.edgeid        = fmesh.variables['indexToEdgeID']
            self.vertexid      = fmesh.variables['indexToVertexID']
            if self.on_sphere:
                self.xcell     = xr.ufuncs.degrees(fmesh.variables['lonCell'])
                self.ycell     = xr.ufuncs.degrees(fmesh.variables['latCell'])
                self.xedge     = xr.ufuncs.degrees(fmesh.variables['lonEdge'])
                self.yedge     = xr.ufuncs.degrees(fmesh.variables['latEdge'])
                self.xvertex   = xr.ufuncs.degrees(fmesh.variables['lonVertex'])
                self.yvertex   = xr.ufuncs.degrees(fmesh.variables['latVertex'])
            else:
                self.xcell     = fmesh.variables['xCell']
                self.ycell     = fmesh.variables['yCell']
                self.xedge     = fmesh.variables['xEdge']
                self.yedge     = fmesh.variables['yEdge']
                self.xvertex   = fmesh.variables['xVertex']
                self.yvertex   = fmesh.variables['yVertex']
            self.acell         = fmesh.variables['areaCell']
            self.adual         = fmesh.variables['areaTriangle']
            self.nedges_cell   = fmesh.variables['nEdgesOnCell']
            self.edges_cell    = fmesh.variables['edgesOnCell']
            self.vertices_cell = fmesh.variables['verticesOnCell']
            self.dc_edge       = fmesh.variables['dcEdge']
            self.dv_edge       = fmesh.variables['dvEdge']
            self.cells_edge    = fmesh.variables['cellsOnEdge']
            self.cells_vertex  = fmesh.variables['cellsOnVertex']
            self.edges_vertex  = fmesh.variables['edgesOnVertex']

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>10s}: {:s}'.format('name', self.name))
        for attr in ['ncells', 'nedges', 'nvertices']:
            summary.append('{:>10s}: {:d}'.format(attr, getattr(self, attr)))
        summary.append('{:>10s}: {}'.format('on sphere',self.on_sphere))
        return '\n'.join(summary)


#--------------------------------
# MPASOMap
#--------------------------------

class MPASOMap:
    """A data type describing a map of MPAS-Ocean field

    """

    def __init__(
            self,
            data = None,
            name = '',
            units = '',
            lon = None,
            lat = None,
            mesh = None,
            position = 'cell',
            mask = None,
            ):
        """Initialization of MPASOMap

        :data:      (array like) data array
        :name:      (str) name of data
        :units:     (str) units of data
        :lon:       (array like) longitude array in degrees
        :lat:       (array like) latitude array in degrees
        :mesh:      (MPASMesh) mesh object
        :position:  (str) position of data (cell (default), edge, vertex)
        :mask:      (array like) mask

        """
        self.data = data
        self.name = name
        self.units = units
        self.mesh = mesh
        self.position = position
        self.mask = mask
        if self.mesh is None:
            self.lon = lon
            self.lat = lat
        else:
            assert isinstance(self.mesh, MPASMesh), 'MPASMesh object is required for mesh, got {}'.format(type(self.mesh))
            assert self.mesh.on_sphere, 'Mesh not on sphere'
            if self.position == 'cell':
                self.lon = mesh.xcell
                self.lat = mesh.ycell
            elif self.position == 'vertex':
                self.lon = mesh.xvertex
                self.lat = mesh.yvertex
            else:
                raise ValueError('Position should be \'cell\' (default), or \'vertex\', got \'{:s}\''.format(self.position))

    def __repr__(self):
        """Formatted print

        """
        size = self.data.size
        summary = [str(self.__class__)+' (size={}):'.format(size)]
        summary.append('{:>8s}: '.format('name')+getattr(self, 'name'))
        summary.append('{:>8s}: '.format('units')+getattr(self, 'units'))
        for attr in ['data', 'lon', 'lat']:
            darr = np.asarray(getattr(self, attr))
            if size > 4:
                dataview = '[{:f} {:f} ... {:f} {:f}]'.format(darr[0], darr[1], darr[-2], darr[-1])
            else:
                dataview = str(darr)
            summary.append('{:>8s}: '.format(attr)+dataview)
        if self.mesh is not None:
            summary.append('{:>8s}: '.format('mesh')+self.mesh.name)
        return '\n'.join(summary)

    def __getitem__(self, index):
        """Slicing

        """
        out = copy.copy(self)
        for attr in ['data', 'lon', 'lat']:
            setattr(out, attr, getattr(self, attr)[index])
        return out

    def plot(
            self,
            axis = None,
            region = 'Global',
            levels = None,
            ptype = 'contourf',
            cmap = 'viridis',
            colorbar = True,
            **kwargs,
            ):
        """Plot figure

        :axis:      (matplotlib.axes, optional) axis to plot the figure on
        :region:    (str, optional) region name, Global by default
        :leveles:   (array-like, optional) list of levels
        :ptype:     (str, optional) plot type, contourf by default
        :cmap:      (str, optional) colormap, viridis by default
        :colorbar:  (bool, optional) do not add colorbar if False
        :**kwargs:  (keyword arguments, optional) passed along to the contourf or PatchCollection constructor
        :return:    (mpl_toolkits.basemap.Basemap) figure handle

        """
        # check input
        if self.mesh is None and ptype != 'contourf':
            raise ValueError('Only \'contourf\' plot is supported without a mesh')
        if region == 'Global' and ptype == 'pcolor':
            warnings.warn('\'pcolor\' on \'Global\' region not supported, using \'contourf\' instead...')
            ptype = 'contourf'
        # use curret axis if not specified
        if axis is None:
            axis = plt.gca()
        # basemap
        m = plot_basemap(region=region, axis=axis)
        (lonmin, lonmax, latmin, latmax) = get_region_llrange(region)
        # longitude wrapping
        lon_wrapping = False
        # convert to numpy array
        data = np.asarray(self.data)
        lon  = np.asarray(self.lon)
        lat  = np.asarray(self.lat)
        if self.mask is None:
            # region mask
            if region == 'Global':
                region_mask = np.full(data.shape, True, dtype=bool)
                lon  = np.where(lon < m.lonmin, lon+360., lon)
                lon  = np.where(lon > m.lonmax, lon-360., lon)
            else:
                if lonmax > 360.:
                    lon_wrapping = True
                    dlon_wrapping = lonmax - 360. + 5.
                    region_mask = ((lon >= lonmin) | (lon <= lonmax%360.)) & (lat >= latmin) & (lat <= latmax)
                else:
                    region_mask = (lon >= lonmin) & (lon <= lonmax) & (lat >= latmin) & (lat <= latmax)
            # nan mask
            nan_mask = (~ np.isnan(data))
            # mask
            mask = region_mask & nan_mask
        else:
            mask = self.mask
        # apply mask
        data = data[mask]
        lon  =  lon[mask]
        lat  =  lat[mask]
        # print message
        print('Plotting \'{:s}\' map in the \'{:s}\' ({:d} data points)...'.format(self.name+' ('+self.units+')', region, data.size))
        # manually mapping levels to the colormap if levels is passed in,
        if levels is not None:
            bounds = np.array(levels)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        else:
            norm = None
        # simple plot if mesh is not defined
        if self.mesh is None or ptype == 'contourf':
            xx, yy = m(lon, lat)
            fig = m.contourf(xx, yy, data, tri=True, levels=levels, extend='both',
                        norm=norm, cmap=plt.cm.get_cmap(cmap), **kwargs)
        else:
            if ptype == 'pcolor':
                if self.position == 'cell':
                    vertexid = np.asarray(self.mesh.vertexid)
                    xvertex = np.asarray(self.mesh.xvertex)
                    if lon_wrapping:
                        xvertex = np.where(xvertex < dlon_wrapping, xvertex+360., xvertex)
                    yvertex = np.asarray(self.mesh.yvertex)
                    nedges_cell = np.asarray(self.mesh.nedges_cell[mask])
                    vertices_cell = np.asarray(self.mesh.vertices_cell[mask,:])
                    xx, yy = m(xvertex, yvertex)
                    fig = ug_pcolor_cell(axis=m.ax, data=data,
                            vertexid=vertexid, xvertex=xx, yvertex=yy,
                            nedges_cell=nedges_cell, vertices_cell=vertices_cell,
                            linewidth=0.1, norm=norm, cmap=plt.cm.get_cmap(cmap),
                            **kwargs)
                else: # self.position == 'vertex'
                    cellid = np.asarray(self.mesh.cellid)
                    xcell = np.asarray(self.mesh.xcell)
                    if lon_wrapping:
                        xcell = np.where(xcell < dlon_wrapping, xcell+360., xcell)
                    ycell = np.asarray(self.mesh.ycell)
                    cells_vertex = np.asarray(self.mesh.cells_vertex[mask,:])
                    xx, yy = m(xcell, ycell)
                    fig = ug_pcolor_vertex(axis=m.ax, data=data,
                            cellid=cellid, xcell=xx, ycell=yy,
                            cells_vertex=cells_vertex,
                            linewidth=0.1, norm=norm, cmap=plt.cm.get_cmap(cmap),
                            **kwargs)
            else:
                raise ValueError('Plot type \'{:s}\' not supported'.format(ptype))
        # add colorbar
        if colorbar:
            cb = m.colorbar(fig, ax=axis)
            cb.set_label('{} ({})'.format(self.name, self.units))
            cb.formatter.set_powerlimits((-4, 4))
            cb.update_ticks()
        return m

