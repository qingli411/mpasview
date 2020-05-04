#--------------------------------
# MPAS data type
#--------------------------------

import os
import copy
import warnings
import textwrap
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .plot import *
from .utils import *

#--------------------------------
# MPASMesh
#--------------------------------

class MPASMesh:

    """A data type for MPAS mesh"""

    def __init__(
            self,
            name = '',
            filepath = '',
            ):
        """Initialization

        :name:      (str) name of the mesh
        :filepath:  (str) path of the MPAS-Ocean mesh file

        """
        assert os.path.isfile(filepath), 'Please set the path of the MPAS mesh file.'
        self.name = name
        self.filepath = filepath

        with xr.open_dataset(self.filepath) as fmesh:
            if fmesh.attrs['on_a_sphere'] == 'YES':
                self.on_sphere = True
            else:
                self.on_sphere = False
            if 'is_periodic' in fmesh.attrs:
                if fmesh.attrs['is_periodic'] == 'YES':
                    self.is_periodic = True
                    self.xperiod = fmesh.attrs['x_period']
                    self.yperiod = fmesh.attrs['y_period']
            else:
                self.is_periodic = False
            self.maxedges_cell = fmesh.dims['maxEdges']
            self.vertexdegree  = fmesh.dims['vertexDegree']
            self.ncells        = fmesh.dims['nCells']
            self.nedges        = fmesh.dims['nEdges']
            self.nvertices     = fmesh.dims['nVertices']
            self.cellid        = fmesh.variables['indexToCellID'].values
            self.edgeid        = fmesh.variables['indexToEdgeID'].values
            self.vertexid      = fmesh.variables['indexToVertexID'].values
            if self.on_sphere:
                self.xcell     = np.degrees(fmesh.variables['lonCell'].values)
                self.ycell     = np.degrees(fmesh.variables['latCell'].values)
                self.xedge     = np.degrees(fmesh.variables['lonEdge'].values)
                self.yedge     = np.degrees(fmesh.variables['latEdge'].values)
                self.xvertex   = np.degrees(fmesh.variables['lonVertex'].values)
                self.yvertex   = np.degrees(fmesh.variables['latVertex'].values)
            else:
                self.xcell     = fmesh.variables['xCell'].values
                self.ycell     = fmesh.variables['yCell'].values
                self.xedge     = fmesh.variables['xEdge'].values
                self.yedge     = fmesh.variables['yEdge'].values
                self.xvertex   = fmesh.variables['xVertex'].values
                self.yvertex   = fmesh.variables['yVertex'].values
            self.acell         = fmesh.variables['areaCell'].values
            self.adual         = fmesh.variables['areaTriangle'].values
            self.nedges_cell   = fmesh.variables['nEdgesOnCell'].values
            self.cells_cell    = fmesh.variables['cellsOnCell'].values
            self.edges_cell    = fmesh.variables['edgesOnCell'].values
            self.vertices_cell = fmesh.variables['verticesOnCell'].values
            self.dc_edge       = fmesh.variables['dcEdge'].values
            self.dv_edge       = fmesh.variables['dvEdge'].values
            self.cells_edge    = fmesh.variables['cellsOnEdge'].values
            self.vertices_edge = fmesh.variables['verticesOnEdge'].values
            self.cells_vertex  = fmesh.variables['cellsOnVertex'].values
            self.edges_vertex  = fmesh.variables['edgesOnVertex'].values
        self.edge_sign_cell = None
        self.edge_sign_vertex = None

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>12s}: {:s}'.format('name', self.name))
        for attr in ['ncells', 'nedges', 'nvertices']:
            summary.append('{:>12s}: {:d}'.format(attr, getattr(self, attr)))
        summary.append('{:>12s}: {}'.format('on sphere',self.on_sphere))
        if not self.on_sphere:
            summary.append('{:>12s}: {}'.format('is periodic',self.is_periodic))
            summary.append('{:>12s}: {}'.format('x period',self.xperiod))
            summary.append('{:>12s}: {}'.format('y period',self.yperiod))
        return '\n'.join(summary)

    def get_edge_sign_on_cell(self):
        """Get the sign of edges on cells

        """
        # only compute the sign of edges if not already computed
        if self.edge_sign_cell is None:
            self.edge_sign_cell = get_edge_sign_on_cell(
                    cellid = self.cellid,
                    nedges_cell = self.nedges_cell,
                    edges_cell = self.edges_cell,
                    cells_edge = self.cells_edge,
                    )

    def get_edge_sign_on_vertex(self):
        """Get the sign of edges on vertices

        """
        # only compute the sign of edges if not already computed
        if self.edge_sign_vertex is None:
            self.edge_sign_vertex = get_edge_sign_on_vertex(
                    vertexid = self.vertexid,
                    edges_vertex = self.edges_vertex,
                    vertices_edge = self.vertices_edge,
                    )

    def get_shortest_path_edge(
            self,
            xP0,
            yP0,
            xP1,
            yP1,
            npoint_ref=1,
            debug_info=False,
            ):
        """Get the shorted path that connects two endpoints (vertices).

        :xP0: (float) x-coordinate of endpoint 0
        :yP0: (float) y-coordinate of endpoint 0
        :xP1: (float) x-coordinate of endpoint 1
        :yP1: (float) y-coordinate of endpoint 1
        :npoint_ref: (int, optional) number of reference points along the straight line or great circle (on a sphere)
        :debug_info: (bool, optional) print out additional debug information if True

        """
        # find indices of endpoints
        idxP0 = get_index_xy(xP0, yP0, self.xvertex, self.yvertex)
        idxP1 = get_index_xy(xP1, yP1, self.xvertex, self.yvertex)
        print('Vertex closest to P0: {:8.5f} {:8.5f}'.format(self.xvertex[idxP0], self.yvertex[idxP0]))
        print('Vertex closest to P1: {:8.5f} {:8.5f}'.format(self.xvertex[idxP1], self.yvertex[idxP1]))
        # find reference points
        x_ref, y_ref = gc_interpolate(self.xvertex[idxP0], self.yvertex[idxP0],
                self.xvertex[idxP1], self.yvertex[idxP1], npoint_ref+2)
        if self.on_sphere:
            x_ref = np.mod(x_ref[1:-1], 360)
            y_ref = np.mod(y_ref[1:-1], 360)
        # initialize an empty path
        out = EdgePath()
        # loop over reference points, find the path between these points
        idx_sp0 = idxP0
        for i in np.arange(npoint_ref):
            idx_vertex = np.minimum(i,1)
            idx_sp1 = get_index_xy(x_ref[i], y_ref[i], self.xvertex, self.yvertex)
            print(' - Vertex closest to RefP{:d}: {:8.5f} {:8.5f}'.format(i+1, self.xvertex[idx_sp1], self.yvertex[idx_sp1]))
            out_i = get_path_edge(idx_sp0, idx_sp1,
                    self.vertexid, self.xvertex, self.yvertex,
                    self.edgeid, self.xedge, self.yedge,
                    self.edges_vertex, self.vertices_edge,
                    self.on_sphere, debug_info)
            out = out + out_i
            idx_sp0 = idx_sp1
        # last path, start from end points P1
        out_n = get_path_edge(idxP1, idx_sp1,
                self.vertexid, self.xvertex, self.yvertex,
                self.edgeid, self.xedge, self.yedge,
                self.edges_vertex, self.vertices_edge,
                self.on_sphere, debug_info)
        out = out + out_n.reverse()
        return out

    def get_shortest_path_cell(
            self,
            xP0,
            yP0,
            xP1,
            yP1,
            npoint_ref=1,
            debug_info=False,
            ):
        """Get the shorted path that connects two endpoints (cell centers).

        :xP0: (float) x-coordinate of endpoint 0
        :yP0: (float) y-coordinate of endpoint 0
        :xP1: (float) x-coordinate of endpoint 1
        :yP1: (float) y-coordinate of endpoint 1
        :npoint_ref: (int, optional) number of reference points along the straight line or great circle (on a sphere)
        :debug_info: (bool, optional) print out additional debug information if True

        """
        # find indices of endpoints
        idxP0 = get_index_xy(xP0, yP0, self.xcell, self.ycell)
        idxP1 = get_index_xy(xP1, yP1, self.xcell, self.ycell)
        print('Cell closest to P0: {:8.5f} {:8.5f}'.format(self.xcell[idxP0], self.ycell[idxP0]))
        print('Cell closest to P1: {:8.5f} {:8.5f}'.format(self.xcell[idxP1], self.ycell[idxP1]))
        # find reference points
        x_ref, y_ref = gc_interpolate(self.xcell[idxP0], self.ycell[idxP0],
                self.xcell[idxP1], self.ycell[idxP1], npoint_ref+2)
        if self.on_sphere:
            x_ref = np.mod(x_ref[1:-1], 360)
            y_ref = np.mod(y_ref[1:-1], 360)
        # initialize an empty path
        out = CellPath()
        # loop over reference points, find the path between these points
        idx_sp0 = idxP0
        for i in np.arange(npoint_ref):
            idx_cell = np.minimum(i,1)
            idx_sp1 = get_index_xy(x_ref[i], y_ref[i], self.xcell, self.ycell)
            print(' - Cell closest to RefP{:d}: {:8.5f} {:8.5f}'.format(i+1, self.xcell[idx_sp1], self.ycell[idx_sp1]))
            out_i = get_path_cell(idx_sp0, idx_sp1,
                    self.cellid, self.xcell, self.ycell,
                    self.cells_cell, self.nedges_cell,
                    self.on_sphere, debug_info)
            out = out + out_i
            idx_sp0 = idx_sp1
        # last path, start from end points P1
        out_n = get_path_cell(idxP1, idx_sp1,
                self.cellid, self.xcell, self.ycell,
                self.cells_cell, self.nedges_cell,
                self.on_sphere, debug_info)
        out = out + out_n.reverse()
        return out

#--------------------------------
# MPASOData
#--------------------------------

class MPASOData:

    """A data type for MPAS-Ocean output data

    """

    def __init__(
            self,
            filepath = '',
            filepath_mesh = '',
            year_ref = 2000,
            ):
        """Initialization

        :filepath:      (str) path of the MPAS-Ocean data file
        :filepath_mesh: (str) path of the corresponding mesh file
        :year_ref: (int) reference year for time

        """
        assert os.path.isfile(filepath), 'Please set the path of the MPAS-Ocean data file.'
        assert os.path.isfile(filepath_mesh), 'Please set the path of the corresponding mesh file.'
        self._filepath = filepath
        self._filepath_mesh = filepath_mesh
        self._year_ref = year_ref
        base_mesh = os.path.basename(filepath_mesh)
        name_mesh = os.path.splitext(base_mesh)[0]
        self.mesh = MPASMesh(name=name_mesh, filepath=filepath_mesh)
        self.time = self.load_time()
        self.depth = self.load_depth()
        self.dataset = self.load_dataset()

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>12s}: {:s}'.format('data path', self._filepath))
        summary.append('{:>12s}: {:s}'.format('mesh path', self._filepath))
        summary.append('{:>12s}: [{}...{}]'.format('time', self.time[0], self.time[-1]))
        summary.append('{:>12s}: [{}...{}]'.format('depth', self.depth.values[0], self.depth.values[-1]))
        return '\n'.join(summary)

    def load_time(
            self,
            ):
        """Load the time dimension and convert to datetime64

        :return:   (datetime64) time

        """
        with xr.open_dataset(self._filepath) as fdata:
            # load time
            xtime = fdata['xtime'].astype(str)
            time_str = [x.strip() for x in xtime.values]
            if int(time_str[0][:4]) < 1970:
                time_str = ['{:04d}'.format(int(s[:4])+self._year_ref)+s[4:] for s in time_str]
            time = pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S')
        return time

    def load_depth(self):
        """Load depth dimension at cell centers

        :return:  (xr.DataArray) depth

        """
        with xr.open_dataset(self._filepath_mesh) as fmesh:
            if 'refZMid' in fmesh.variables.keys():
                z = fmesh.data_vars['refZMid'].values
            elif 'refBottomDepth' in fmesh.variables.keys():
                bottom_depth = fmesh.data_vars['refBottomDepth'].values
                z = np.zeros_like(bottom_depth)
                z[0] = -0.5*bottom_depth[0]
                z[1:] = -0.5*(bottom_depth[0:-1]+bottom_depth[1:])
            else:
                raise LookupError('Neither \'refZMid\' or \'refBottomDepth\' is found.')
            depth = xr.DataArray(
                    z,
                    dims=('nVertLevels'),
                    coords={'nVertLevels': z},
                    attrs={'units': 'm', 'long_name': 'depth'},
                    )
        return depth

    def load_dataset(
            self,
            ):
        """Load MPASOdata as an xarray.Dataset

        :return:   (xarray.Dataset) data set

        """
        with xr.open_dataset(self._filepath) as fdata:
            out = fdata.assign_coords({
                'Time': self.time,
                'nVertLevels': np.arange(fdata.dims['nVertLevels']),
                'nCells': np.arange(fdata.dims['nCells']),
                'nEdges': np.arange(fdata.dims['nEdges']),
                'nVertices': np.arange(fdata.dims['nVertices']),
                })
            if 'nVertLevelsLES' in fdata.dims:
                out = out.assign_coords({
                    'nVertLevelsLES': np.arange(fdata.dims['nVertLevelsLES']),
                    })
            return out

    def load_variable_profile(
            self,
            varname,
            ):
        """Load a variable from MPASOData as an xarray.DataArray and assign vertical coordinate
           (Currently assuming the vertical coordinate does not vary in time)

        :varname: (str) variable name
        :returns: (xarray.DataArray) variable

        """
        var = self.dataset.data_vars[varname]
        if 'nVertLevels' not in var.dims and 'nVertLevelsLES' not in var.dims:
            raise LookupError('\'{}\' is not a profile variables')
        if 'LES' in varname:
            # LES variables with different vertical levels
            with xr.open_dataset(self._filepath_mesh) as fmesh:
                z = fmesh.data_vars['zLES'].values[0,0,:]
                depth = xr.DataArray(
                        z,
                        dims=('nVertLevelsLES'),
                        coords={'nVertLevelsLES': z},
                        attrs={'units': 'm', 'long_name': 'depth'},
                        )
            var = var.assign_coords({'nVertLevelsLES': depth})
        else:
            var = var.assign_coords({'nVertLevels': self.depth})
        return var.transpose()

    def load_variable_domain(
            self,
            varname,
            itime = None,
            idepth = None,
            time = None,
            depth = None,
            ):
        """Load a variable from MPASOData as an MPASODomain

        :varname: (str) variable name
        :itime:   (int) index of time dimension
        :idepth:  (int) index of depth dimension
        :time:    (numpy.datetime64) value of time dimension
        :depth:   (float) value of depth dimension
        :returns: (xarray.DataArray) variable

        """
        # default values
        if itime is None and time is None:
            itime = -1
        if idepth is None and depth is None:
            idepth = 0
        # load variable
        try:
            var = self.load_variable_profile(varname)
        except LookupError:
            var = self.dataset.data_vars[varname]
        # check position
        if 'nCells' in var.dims:
            position = 'cell'
        elif 'nVertices' in var.dims:
            position = 'vertex'
        else:
            raise LookupError('\'{}\' is not a domain variable'.format(varname))
        print('Loading MPASODomain of \'{}\''.format(varname))
        # check time dimension
        if 'Time' not in var.dims:
            data_s2 = var
        else:
            if isinstance(itime, int):
                data_s1 = var.isel(Time=itime)
            elif time is not None:
                data_s1 = var.sel(Time=time)
            else:
                raise TypeError('Either \'itime\' in \'int\' or time is required')
            print('  time = {}'.format(data_s1.coords['Time'].values))
            # check depth dimension
            ndim = len(data_s1.dims)
            if ndim == 1:
                data_s2 = data_s1
            elif ndim == 2 and 'nVertLevels' in data_s1.dims:
                if isinstance(idepth, int):
                    data_s2 = data_s1.isel(nVertLevels=idepth)
                elif depth is not None:
                    data_s2 = data_s1.sel(nVertLevels=depth)
                else:
                    raise TypeError('Either \'idepth\' in \'int\' or depth is required')
                print(' detph = {} ({})'.format(
                    data_s2.coords['nVertLevels'].values,
                    data_s2.coords['nVertLevels'].attrs['units']))
            elif ndim == 2 and 'nVertLevelsLES' in data_s1.dims:
                if isinstance(idepth, int):
                    data_s2 = data_s1.isel(nVertLevelsLES=idepth)
                elif depth is not None:
                    data_s2 = data_s1.sel(nVertLevelsLES=depth)
                else:
                    raise TypeError('Either \'idepth\' in \'int\' or depth is required')
                print(' detph = {} ({})'.format(
                    data_s2.coords['nVertLevelsLES'].values,
                    data_s2.coords['nVertLevelsLES'].attrs['units']))
            else:
                raise LookupError('\'{}\' is not a domain variable')
        # create MPASOMap
        out = MPASODomain(
                data = data_s2.values,
                name = var.attrs['long_name'],
                units = var.attrs['units'],
                mesh = self.mesh,
                )
        return out

#--------------------------------
# MPASOMap
#--------------------------------

class MPASOMap:
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
            position = 'cell',
            mask = None,
            ):
        """Initialization

        :data:      (array like) data array
        :name:      (str) name of data
        :units:     (str) units of data
        :lon:       (array like) longitude array in degrees
        :lat:       (array like) latitude array in degrees
        :mesh:      (MPASMesh) mesh object
        :position:  (str) position of data (cell (default), or vertex)
        :mask:      (array like) mask

        """
        self.data = np.asarray(data)
        self.name = name
        self.units = units
        self.mesh = mesh
        self.position = position
        self.mask = mask
        if self.mesh is None:
            self.lon = np.asarray(lon)
            self.lat = np.asarray(lat)
        else:
            assert isinstance(self.mesh, MPASMesh), 'MPASMesh object is required for mesh, got {}'.format(type(self.mesh))
            assert self.mesh.on_sphere, 'Mesh not on sphere, use MPASODomain'
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
            darr = getattr(self, attr)
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
        (lonmin, lonmax, latmin, latmax) = get_info_region(region)
        # longitude wrapping
        lon_wrapping = False
        # convert to numpy array
        data = self.data
        lon  = self.lon
        lat  = self.lat
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
            nan_mask = np.isnan(data)
            # mask
            mask = region_mask & (~ nan_mask)
            fill_mask = region_mask & nan_mask
        else:
            mask = self.mask
        # apply mask
        data = self.data[mask]
        lon  =  lon[mask]
        lat  =  lat[mask]
        lon_fill = self.lon[fill_mask]
        lat_fill = self.lat[fill_mask]
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
            # contourf plot
            if lon_wrapping:
                lon = np.where(lon < dlon_wrapping, lon+360., lon)
            xx, yy = m(lon, lat)
            fig = m.contourf(xx, yy, data, tri=True, levels=levels, extend='both',
                        norm=norm, cmap=plt.cm.get_cmap(cmap), **kwargs)
            # fill nan
            if self.position == 'cell':
                vertexid = self.mesh.vertexid
                xvertex = self.mesh.xvertex
                if lon_wrapping:
                    xvertex = np.where(xvertex < dlon_wrapping, xvertex+360., xvertex)
                yvertex = self.mesh.yvertex
                nedges_cell = self.mesh.nedges_cell[fill_mask]
                vertices_cell = self.mesh.vertices_cell[fill_mask,:]
                xx, yy = m(xvertex, yvertex)
                ug_pcolor_cell(axis=m.ax,
                        vertexid=vertexid, xvertex=xx, yvertex=yy,
                        nedges_cell=nedges_cell, vertices_cell=vertices_cell,
                        linewidth=0.8, facecolors='lightgray', edgecolors='lightgray', alpha=1.0)
            else: # self.position == 'vertex'
                cellid = self.mesh.cellid
                xcell = self.mesh.xcell
                if lon_wrapping:
                    xcell = np.where(xcell < dlon_wrapping, xcell+360., xcell)
                ycell = self.mesh.ycell
                cells_vertex = self.mesh.cells_vertex[fill_mask,:]
                xx, yy = m(xcell, ycell)
                ug_pcolor_vertex(axis=m.ax, data=data,
                        cellid=cellid, xcell=xx, ycell=yy,
                        cells_vertex=cells_vertex,
                        linewidth=0.8, facecolors='lightgray', edgecolors='lightgray', alpha=1.0)
        else:
            if ptype == 'pcolor':
                if self.position == 'cell':
                    vertexid = self.mesh.vertexid
                    xvertex = self.mesh.xvertex
                    if lon_wrapping:
                        xvertex = np.where(xvertex < dlon_wrapping, xvertex+360., xvertex)
                    yvertex = self.mesh.yvertex
                    nedges_cell = self.mesh.nedges_cell[mask]
                    vertices_cell = self.mesh.vertices_cell[mask,:]
                    xx, yy = m(xvertex, yvertex)
                    fig = ug_pcolor_cell(axis=m.ax, data=data,
                            vertexid=vertexid, xvertex=xx, yvertex=yy,
                            nedges_cell=nedges_cell, vertices_cell=vertices_cell,
                            linewidth=0.1, norm=norm, cmap=plt.cm.get_cmap(cmap),
                            **kwargs)
                else: # self.position == 'vertex'
                    cellid = self.mesh.cellid
                    xcell = self.mesh.xcell
                    if lon_wrapping:
                        xcell = np.where(xcell < dlon_wrapping, xcell+360., xcell)
                    ycell = self.mesh.ycell
                    cells_vertex = self.mesh.cells_vertex[mask,:]
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
            cb_label = '\n'.join(textwrap.wrap('{} ({})'.format(self.name, self.units), 30))
            cb.set_label(cb_label)
            cb.formatter.set_powerlimits((-4, 4))
            cb.update_ticks()
        return m

#--------------------------------
# MPASODomain
#--------------------------------

class MPASODomain:
    """A data type describing a horizontal domain of MPAS-Ocean field

    """

    def __init__(
            self,
            data = np.nan,
            name = '',
            units = '',
            x = np.nan,
            y = np.nan,
            mesh = None,
            position = 'cell',
            mask = None,
            ):
        """Initialization

        :data:      (array like) data array
        :name:      (str) name of data
        :units:     (str) units of data
        :x:         (array like) x-coordinate
        :y:         (array like) y-coordinate
        :mesh:      (MPASMesh) mesh object
        :position:  (str) position of data (cell (default), or vertex)
        :mask:      (array like) mask

        """

        self.data = np.asarray(data)
        self.name = name
        self.units = units
        self.mesh = mesh
        self.position = position
        self.mask = mask
        if self.mesh is None:
            self.x = np.asarray(x)
            self.y = np.asarray(y)
        else:
            assert isinstance(self.mesh, MPASMesh), 'MPASMesh object is required for mesh, got {}'.format(type(self.mesh))
            assert not self.mesh.on_sphere, 'Mesh on sphere, use MPASOMap'
            if self.position == 'cell':
                self.x = mesh.xcell
                self.y = mesh.ycell
            elif self.position == 'vertex':
                self.x = mesh.xvertex
                self.y = mesh.yvertex
            else:
                raise ValueError('Position should be \'cell\' (default), or \'vertex\', got \'{:s}\''.format(self.position))

    def __repr__(self):
        """Formatted print

        """
        size = self.data.size
        summary = [str(self.__class__)+' (size={}):'.format(size)]
        summary.append('{:>8s}: '.format('name')+getattr(self, 'name'))
        summary.append('{:>8s}: '.format('units')+getattr(self, 'units'))
        for attr in ['data', 'x', 'y']:
            darr = getattr(self, attr)
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
        for attr in ['data', 'x', 'y']:
            setattr(out, attr, getattr(self, attr)[index])
        return out

    def plot(
            self,
            axis = None,
            levels = None,
            ptype = 'contourf',
            cmap = 'viridis',
            colorbar = True,
            **kwargs,
            ):
        """Plot figure

        :axis:      (matplotlib.axes, optional) axis to plot the figure on
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
        # use curret axis if not specified
        if axis is None:
            axis = plt.gca()
        # apply mask
        if self.mask is not None:
            data = self.data[self.mask]
            x    =    self.x[self.mask]
            y    =    self.y[self.mask]
        else:
            data = self.data
            x    = self.x
            y    = self.y
        # print message
        print('Plotting \'{:s}\' on x-y domain ({:d} data points)...'.format(self.name+' ('+self.units+')', data.size))
        # manually mapping levels to the colormap if levels is passed in,
        if levels is not None:
            bounds = np.array(levels)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        else:
            norm = None
        # simple plot if mesh is not defined
        if self.mesh is None or ptype == 'contourf':
            fig = axis.tricontourf(x, y, data, levels=levels, extend='both',
                        norm=norm, cmap=plt.cm.get_cmap(cmap), **kwargs)
        else:
            if ptype == 'pcolor':
                if self.position == 'cell':
                    vertexid = self.mesh.vertexid
                    if self.mask is not None:
                        nedges_cell = self.mesh.nedges_cell[self.mask]
                        vertices_cell = self.mesh.vertices_cell[self.mask,:]
                    else:
                        nedges_cell = self.mesh.nedges_cell
                        vertices_cell = self.mesh.vertices_cell
                    if self.mesh.is_periodic:
                        fig = ug_pcolor_cell_periodic(axis=axis, data=data,
                                xperiod=self.mesh.xperiod,
                                yperiod=self.mesh.yperiod,
                                vertexid=self.mesh.vertexid,
                                xvertex=self.mesh.xvertex,
                                yvertex=self.mesh.yvertex,
                                xcell=self.mesh.xcell,
                                ycell=self.mesh.ycell,
                                dv_edge=self.mesh.dv_edge,
                                nedges_cell=nedges_cell,
                                vertices_cell=vertices_cell,
                                norm=norm, cmap=plt.cm.get_cmap(cmap),
                                **kwargs)
                    else:
                        fig = ug_pcolor_cell(axis=axis, data=data,
                                vertexid=self.mesh.vertexid,
                                xvertex=self.mesh.xvertex,
                                yvertex=self.mesh.yvertex,
                                nedges_cell=nedges_cell,
                                vertices_cell=vertices_cell,
                                norm=norm, cmap=plt.cm.get_cmap(cmap),
                                **kwargs)
                else: # self.position == 'vertex'
                    cellid = self.mesh.cellid
                    if self.mask is not None:
                        cells_vertex = self.mesh.cells_vertex[self.mask,:]
                    else:
                        cells_vertex = self.mesh.cells_vertex
                    if self.mesh.is_periodic:
                        fig = ug_pcolor_vertex_periodic(axis=axis, data=data,
                                xperiod=self.mesh.xperiod,
                                yperiod=self.mesh.yperiod,
                                cellid=self.mesh.cellid,
                                xvertex=self.mesh.xvertex,
                                yvertex=self.mesh.yvertex,
                                xcell=self.mesh.xcell,
                                ycell=self.mesh.ycell,
                                dv_edge=self.mesh.dv_edge,
                                cells_vertex=cells_vertex,
                                norm=norm, cmap=plt.cm.get_cmap(cmap),
                                **kwargs)
                    else:
                        fig = ug_pcolor_vertex(axis=axis, data=data,
                                cellid=self.mesh.cellid,
                                xcell=self.mesh.xcell,
                                ycell=self.mesh.ycell,
                                cells_vertex=cells_vertex,
                                norm=norm, cmap=plt.cm.get_cmap(cmap),
                                **kwargs)
            else:
                raise ValueError('Plot type \'{:s}\' not supported'.format(ptype))
        # add colorbar
        if colorbar:
            cb = plt.colorbar(fig, ax=axis)
            cb_label = '\n'.join(textwrap.wrap('{} ({})'.format(self.name, self.units), 30))
            cb.set_label(cb_label)
            cb.formatter.set_powerlimits((-4, 4))
            cb.update_ticks()
        return fig

