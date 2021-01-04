#--------------------------------
# Untilities
#--------------------------------

import numpy as np
from scipy import spatial
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from .plot import *

#--------------------------------
# Path
#--------------------------------

class CellPath:

    """CellPath object defined by connected cells

    """

    def __init__(
            self,
            icell = [],
            xcell = [],
            ycell = [],
            ):
        """Initialization

        :icell:     (array-like, optional) index of cells
        :xcell:     (array-like, optional) x-coordinate of cells
        :ycell:     (array-like, optional) y-coordinate of cells

        """

        self.icell = list(icell)
        self.xcell = list(xcell)
        self.ycell = list(ycell)

    def __add__(self, other):
        """Connect two paths

        """
        if len(self.xcell) > 0 and len(other.xcell) > 0:
            assert self.xcell[-1] == other.xcell[0], 'Cannot connect the two paths due to inconsitant x-coordinate of the end points'
            assert self.ycell[-1] == other.ycell[0], 'Cannot connect the two paths due to inconsitant x-coordinate of the end points'
        for attr in self.__dict__.keys():
            attr_val = getattr(self, attr)
            if isinstance(attr_val, list):
                attr_val.extend(getattr(other, attr))
                setattr(self, attr, attr_val)
        return self

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>10s}: {:d}'.format('ncells', len(self.xcell)))
        summary.append(' P0({} {}) -> P1({} {})'.format(self.xcell[0], self.ycell[0], self.xcell[-1], self.ycell[-1]))
        return '\n'.join(summary)

    def reverse(self):
        """Reverse the order of path

        """
        for attr in self.__dict__.keys():
            attr_val = getattr(self, attr)
            if isinstance(attr_val, list):
                attr_val.reverse()
                setattr(self, attr, attr_val)
        return self

    def project_cell_center(self, axis, **kwargs):
        """Project the cell centers along a path on axis

        :axis:  (matplotlib.axes or cartopy.mpl.geoaxes.GeoAxes) axis to project the path

        """
        if isinstance(axis, GeoAxes):
            xx, yy, _ = axis.projection.transform_points(ccrs.PlateCarree(),
                        np.array(self.xcell), np.array(self.ycell)).T
        else:
            xx, yy = self.xcell, self.ycell
        out = axis.plot(xx, yy, **kwargs)
        return out

    def project_cell_filled(self, axis, mesh, **kwargs):
        """Project the cells along a path on axis

        :axis:  (matplotlib.axes or cartopy.mpl.geoaxes.GeoAxes) axis to project the path
        :mesh:  (mpasview.data.MPASMesh) mesh object

        """
        nedges_cell = mesh.nedges_cell[self.icell]
        vertices_cell = mesh.vertices_cell[self.icell,:]
        if isinstance(axis, GeoAxes):
            xx, yy, _ = axis.projection.transform_points(ccrs.PlateCarree(),
                        np.array(mesh.xvertex), np.array(mesh.yvertex)).T
        else:
            xx, yy = mesh.xvertex, mesh.yvertex
        out = ug_pcolor_cell(axis=axis,
                vertexid=mesh.vertexid, xvertex=xx, yvertex=yy,
                nedges_cell=nedges_cell, vertices_cell=vertices_cell,
                **kwargs)
        return out

class EdgePath:

    """Path object defined by connected edges

    """

    def __init__(
            self,
            ivertex = [],
            xvertex = [],
            yvertex = [],
            iedge = [],
            xedge = [],
            yedge = [],
            sign_edges = [],
            ):
        """Initialization

        :ivertex:     (array-like, optional) index of vertices
        :xvertex:     (array-like, optional) x-coordinate of vertices
        :yvertex:     (array-like, optional) y-coordinate of vertices
        :iedge:       (array-like, optional) index of edges
        :xedge:       (array-like, optional) x-coordinate of edges
        :yedge:       (array-like, optional) y-coordinate of edges
        :sign_edges:  (array-like, optional) sign of edges

        """

        self.ivertex = list(ivertex)
        self.xvertex = list(xvertex)
        self.yvertex = list(yvertex)
        self.iedge = list(iedge)
        self.xedge = list(xedge)
        self.yedge = list(yedge)
        self.sign_edges = list(sign_edges)

    def __add__(self, other):
        """Connect two paths

        """
        if len(self.xvertex) == 0:
            idx_v = 0
        else:
            assert self.xvertex[-1] == other.xvertex[0], 'Cannot connect the two paths due to inconsitant x-coordinate of the end points'
            assert self.yvertex[-1] == other.yvertex[0], 'Cannot connect the two paths due to inconsitant x-coordinate of the end points'
            idx_v = 1
        for attr in self.__dict__.keys():
            attr_val = getattr(self, attr)
            if isinstance(attr_val, list):
                if 'vertex' in attr:
                    attr_val.extend(getattr(other, attr)[idx_v:])
                else:
                    attr_val.extend(getattr(other, attr))
                setattr(self, attr, attr_val)
        return self

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>10s}: {:d}'.format('nvertices', len(self.xvertex)))
        summary.append(' P0({} {}) -> P1({} {})'.format(self.xvertex[0], self.yvertex[0], self.xvertex[-1], self.yvertex[-1]))
        return '\n'.join(summary)

    def reverse(self):
        """Reverse the order of path

        """
        for attr in self.__dict__.keys():
            attr_val = getattr(self, attr)
            if isinstance(attr_val, list):
                attr_val.reverse()
                setattr(self, attr, attr_val)
        self.sign_edges = [-1*val for val in self.sign_edges]
        return self

    def project_edge_center(self, axis, s=1, **kwargs):
        """Project the edge centers along a path on axis

        :axis:  (matplotlib.axes or cartopy.mpl.geoaxes.GeoAxes) axis to project the path
        :s:     (str) size of the scatter

        """
        if isinstance(axis, GeoAxes):
            xx, yy, _ = axis.projection.transform_points(ccrs.PlateCarree(),
                        np.array(self.xedge), np.array(self.yedge)).T
        else:
            xx, yy = self.xedge, self.yedge
        out = axis.scatter(xx, yy, s=s, **kwargs)
        return out

    def project_vertex(self, axis, s=1, **kwargs):
        """Project the vertices along a path on axis

        :axis:  (matplotlib.axes or cartopy.mpl.geoaxes.GeoAxes) axis to project the path
        :s:     (str) size of the scatter

        """
        if isinstance(axis, GeoAxes):
            xx, yy, _ = axis.projection.transform_points(ccrs.PlateCarree(),
                        np.array(self.xvertex), np.array(self.yvertex)).T
        else:
            xx, yy = self.xvertex, self.yvertex
        out = axis.scatter(xx, yy, s=s, **kwargs)
        return out

    def project_edge(self, axis, **kwargs):
        """Project the edges along a path on axis

        :axis:  (matplotlib.axes or cartopy.mpl.geoaxes.GeoAxes) axis to project the path

        """
        if isinstance(axis, GeoAxes):
            xx, yy, _ = axis.projection.transform_points(ccrs.PlateCarree(),
                        np.array(self.xvertex), np.array(self.yvertex)).T
        else:
            xx, yy = self.xvertex, self.yvertex
        out = axis.plot(xx, yy, **kwargs)
        return out

#--------------------------------
# Functions on mesh
#--------------------------------
def get_edge_sign_on_cell(
        cidx = None,
        cellid = np.nan,
        nedges_cell = np.nan,
        edges_cell = np.nan,
        cells_edge = np.nan,
        ):
    """Get the sign of edges on cells

    :cellid:        (array-like) cell ID
    :nedges_cell:   (array-like) number of edges on cells
    :edges_cell:    (array-like) edges on cells
    :cells_edge:    (array-like) cells on edges
    :return:        (numpy array) sign of edges on cells

    """
    edge_sign_on_cell = np.zeros_like(edges_cell)
    ncell = edges_cell.shape[0]
    for i in np.arange(ncell):
        for j in np.arange(nedges_cell[i]):
            # TODO assuming the index of the edge is the edge ID - 1
            idx_e = edges_cell[i,j]-1
            if cellid[i] == cells_edge[idx_e, 0]:
                edge_sign_on_cell[i,j] = -1
            else:
                edge_sign_on_cell[i,j] = 1
    return edge_sign_on_cell

def get_edge_sign_on_vertex(
        vertexid = np.nan,
        edges_vertex = np.nan,
        vertices_edge = np.nan,
        ):
    """Get the sign of edges on vertices

    :vertexid:      (array-like) vertex ID
    :edges_vertex:  (array-like) edges on vertices
    :vertices_edge: (array-like) vertices on edges
    :return:        (numpy array) sign of edges on vertices

    """
    edge_sign_on_vertex = np.zeros_like(edges_vertex)
    nvertex = edges_vertex.shape[0]
    for i in np.arange(nvertex):
        for j in np.arange(3):
            # TODO assuming the index of the vertex is the vertex ID - 1
            idx_e = edges_vertex[i,j]-1
            if vertexid[i] == vertices_edge[idx_e, 0]:
                edge_sign_on_vertex[i,j] = -1
            else:
                edge_sign_on_vertex[i,j] = 1
    return edge_sign_on_vertex

def get_path_cell(
        cidx_p0,
        cidx_p1,
        cellid,
        xcell,
        ycell,
        cells_cell,
        nedges_cell,
        on_sphere = True,
        debug_info = False,
        ):
    """Get the path between two endpoints (cell centers) p0 and p1 by connecting the cells

    :cidx_p0:       (int) cell index of p0
    :cidx_p1:       (int) cell index of p1
    :cellid:        (array-like) cell ID
    :xcell:         (array-like) x-coordinate of cells
    :ycell:         (array-like) y-coordinate of cells
    :cells_cell:    (array-like) cells on cells
    :nedges_cell:   (array-like) number of edges on cells
    :on_sphere:     (bool, optional) the mesh is on a sphere if True
    :debug_info:    (bool, optional) print out additional debug information if True

    """
    # initialize arrays
    idx_cells_on_path    = []
    # start from cell P0
    idx_cell_now = cidx_p0
    # record cells on path and the indices
    idx_cells_on_path.append(idx_cell_now)
    if debug_info:
        print('\nCell on path ({:d}): {:8.5f} {:8.5f}'.format(idx_cell_now, xcell[idx_cell_now], ycell[idx_cell_now]))

    # continue if not reached P1
    istep = 0
    while idx_cell_now != cidx_p1:
        # print the step
        if debug_info:
            print('\nStep {:d}'.format(istep))
        # find the indices of the neighboring cells on cell
        cell_arr     = cells_cell[idx_cell_now,:nedges_cell[idx_cell_now]]
        # TODO assuming the index of the cell is the cell ID - 1
        idx_cell_arr = cell_arr-1
        # compute the distance from P1
        dist = []
        idx_tmp = []
        for idx in idx_cell_arr:
            if idx not in idx_cells_on_path:
                xi = xcell[idx]
                yi = ycell[idx]
                if on_sphere:
                    dist.append(gc_distance(xi, yi, xcell[cidx_p1], ycell[cidx_p1]))
                else:
                    dist.append(np.sqrt((xi-xcell[cidx_p1])**2+(yi-ycell[cidx_p1])**2))
                idx_tmp.append(idx)
        # print the location of the neighboring cells
        if debug_info:
            print('\nCells on cell:')
            for i, idx in enumerate(idx_tmp):
                print('   Cell {:d} ({:d}): {:8.5f} {:8.5f} ({:10.4f})'.\
                      format(i, idx, xcell[idx], ycell[idx], dist[i]))
        # choose the cell from the list that is closest to cell P1
        idx_min = np.argmin(dist)
        idx_cell_next = idx_tmp[idx_min]
        # print the cell on path
        if debug_info:
            print('\nCell on path : [Cell {:d} ({:d})] {:8.5f} {:8.5f}'.\
                  format(idx_min, idx_cell_next, xcell[idx_cell_next], ycell[idx_cell_next]))
        # record cell on path and the indices
        idx_cells_on_path.append(idx_cell_next)
        # move to next cell
        idx_cell_now  = idx_cell_next
        # count steps
        istep += 1

    # create a path on MPAS mesh
    i_cell = idx_cells_on_path
    x_cell = xcell[idx_cells_on_path]
    y_cell = ycell[idx_cells_on_path]
    out = CellPath(
            icell=i_cell,
            xcell=x_cell,
            ycell=y_cell,
            )
    return out

def get_path_edge(
        vidx_p0,
        vidx_p1,
        vertexid,
        xvertex,
        yvertex,
        edgeid,
        xedge,
        yedge,
        edges_vertex,
        vertices_edge,
        on_sphere = True,
        debug_info = False,
        ):
    """Get the path between two endpoints (vertices) p0 and p1 by connecting the edges

    :vidx_p0:       (int) vertex index of p0
    :vidx_p1:       (int) vertex index of p1
    :vertexid:      (array-like) vertex ID
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :edgeid:        (array-like) edge ID
    :xedge:         (array-like) x-coordinate of edges
    :yedge:         (array-like) y-coordinate of edges
    :edges_vertex:  (array-like) edges on vertices
    :vertices_edge: (array-like) vertices on edges
    :on_sphere:     (bool, optional) the mesh is on a sphere if True
    :debug_info:    (bool, optional) print out additional debug information if True

    """
    # initialize arrays
    idx_edges_on_path    = []
    idx_vertices_on_path = []
    sign_edges = []
    # start from vertex P0
    idx_vertex_now = vidx_p0
    # record vertices on path and the indices
    idx_vertices_on_path.append(idx_vertex_now)
    if debug_info:
        print('\nVertex on path ({:d}): {:8.5f} {:8.5f}'.format(idx_vertex_now, xvertex[idx_vertex_now], yvertex[idx_vertex_now]))

    # continue if not reached P1
    istep = 0
    while idx_vertex_now != vidx_p1:
        # print the step
        if debug_info:
            print('\nStep {:d}'.format(istep))
        # find the indices of the three edges on vertex
        edge_arr     = edges_vertex[idx_vertex_now,:]
        # TODO assuming the index of the edge is the edge ID - 1
        idx_edge_arr = edge_arr-1
        # compute the distance from P1
        dist = []
        idx_tmp = []
        for idx in idx_edge_arr:
            if idx not in idx_edges_on_path:
                xi = xedge[idx]
                yi = yedge[idx]
                if on_sphere:
                    dist.append(gc_distance(xi, yi, xvertex[vidx_p1], yvertex[vidx_p1]))
                else:
                    dist.append(np.sqrt((xi-xvertex[vidx_p1])**2+(yi-yvertex[vidx_p1])**2))
                idx_tmp.append(idx)
        # print the location of the three edges
        if debug_info:
            print('\nEdges on vertex:')
            for i, idx in enumerate(idx_tmp):
                print('   Edge {:d} ({:d}): {:8.5f} {:8.5f} ({:10.4f})'.\
                      format(i, idx, xedge[idx], yedge[idx], dist[i]))
        # choose the edge from the three that is closest to vertex P1
        idx_min = np.argmin(dist)
        idx_edge_next = idx_tmp[idx_min]
        # print the edge on path
        if debug_info:
            print('\nEdge on path : [Edge {:d} ({:d})] {:8.5f} {:8.5f}'.\
                  format(idx_min, idx_edge_next, xedge[idx_edge_next], yedge[idx_edge_next]))
        # record edges on path and the indices
        idx_edges_on_path.append(idx_edge_next)
        # find the other vertex on this edge
        vertex_arr = vertices_edge[idx_edge_next,:]
        if vertex_arr[0] == vertexid[idx_vertex_now]:
            vertex_next = vertex_arr[1]
            sign_edges.append(-1)
        else:
            vertex_next = vertex_arr[0]
            sign_edges.append(1)
        idx_vertex_next = vertex_next-1
        # record vortices on path and the indices
        idx_vertices_on_path.append(idx_vertex_next)
        if debug_info:
            print('\nVertex on path ({:d}): {:8.5f} {:8.5f}'.\
                  format(idx_vertex_next, xvertex[idx_vertex_next], yvertex[idx_vertex_next]))
        # move to next vertex
        idx_vertex_now  = idx_vertex_next
        # count steps
        istep += 1

    # create a path on MPAS mesh
    i_edge = idx_edges_on_path
    x_edge = xedge[idx_edges_on_path]
    y_edge = yedge[idx_edges_on_path]
    i_vertex = idx_vertices_on_path
    x_vertex = xvertex[idx_vertices_on_path]
    y_vertex = yvertex[idx_vertices_on_path]
    out = EdgePath(
            ivertex=i_vertex,
            xvertex=x_vertex,
            yvertex=y_vertex,
            iedge=i_edge,
            xedge=x_edge,
            yedge=y_edge,
            sign_edges=sign_edges,
            )
    return out

#--------------------------------
# utility
#--------------------------------

def get_index_lonlat(
        loni,
        lati,
        lon_arr,
        lat_arr,
        search_range=5.0,
        ):
    """Get the index of the location (loni, lati) in an array of
       locations (lon_arr, lat_arr)

    :loni:          (float) longitude of target location
    :lati:          (float) latitude of target location
    :lon_arr:       (numpy array) array of longitude
    :lat_arr:       (numpy array) array of latitude
    :search_range:  (float, optional) range of lon and lat for faster search

    """
    lon_mask = (lon_arr>=loni-search_range) & (lon_arr<=loni+search_range)
    lat_mask = (lat_arr>=lati-search_range) & (lat_arr<=lati+search_range)
    lonlat_mask = lon_mask & lat_mask
    lon_sub = lon_arr[lonlat_mask]
    lat_sub = lat_arr[lonlat_mask]
    # scale the distance in the zonal direction by the latitude, centered on
    # the target location
    lon_sub_r = loni + (lon_sub-loni)*np.cos(np.radians(lat_sub))
    pts = np.array([loni,lati])
    tree = spatial.KDTree(list(zip(lon_sub_r, lat_sub)))
    p = tree.query(pts)
    cidx = p[1]
    idx = np.argwhere(lon_arr==lon_sub[cidx])
    if idx.size > 1:
        for i in idx.squeeze():
            if lat_arr[i] == lat_sub[cidx]:
                out = i
                break
    else:
        out = idx[0][0]
    return out

def get_index_xy(
        xi,
        yi,
        x_arr,
        y_arr,
        search_range=5.0,
        ):
    """Get the index of the location (xi, yi) in an array of
       locations (x_arr, y_arr)

    :xi:            (float) x-coordinate of target location
    :yi:            (float) y-coordinate of target location
    :x_arr:         (numpy array) array of x-coordinate
    :y_arr:         (numpy array) array of y-coordinate
    :search_range:  (float, optional) range of x and y for faster search

    """
    x_mask = (x_arr>=xi-search_range) & (x_arr<=xi+search_range)
    y_mask = (y_arr>=yi-search_range) & (y_arr<=yi+search_range)
    xy_mask = x_mask & y_mask
    x_sub = x_arr[xy_mask]
    y_sub = y_arr[xy_mask]
    pts = np.array([xi,yi])
    tree = spatial.KDTree(list(zip(x_sub, y_sub)))
    p = tree.query(pts)
    cidx = p[1]
    idx = np.argwhere(x_arr==x_sub[cidx])
    if idx.size > 1:
        for i in idx.squeeze():
            if y_arr[i] == y_sub[cidx]:
                out = i
                break
    else:
        out = idx[0][0]
    return out

#--------------------------------
# predefined regions and transects
#--------------------------------

def get_info_region(name):
    """Get the range of longitude and latitude of a predefined region.

    :name:      (str) region name
    :returns:   (tuple) region information (lon_min, lon_max, lat_min, lat_max)

    """
    info = {
            'Global': (  0., 360., -90., 90.),
            'Arctic': (  0., 360.,  40., 90.),
            'LabSea': (270., 356.,  36., 75.),
            'TropicalPacific':  (130., 290., -20., 20.),
            'TropicalPacificSmall':  (160., 280., -10., 10.),
            'TropicalAtlantic': (310., 380., -20., 20.),
            }
    if name in info.keys():
        return info.get(name)
    else:
        raise ValueError('Region \'{:s}\' not found.\n'.format(name) \
                + '- Supported region names:\n' \
                + '  ' + ', '.join(info.keys()))

def get_info_transect(name):
    """Get the latitude/longitude of the two end points and the depth of predefiend transects.

    :name:      (str) transect name
    :returns:   (tuple) transect information (lon_P0, lat_P0, lon_P1, lat_P1, depth)

    """
    info = {
            'AR7W':             (304.,  53.5, 312.,  61.,  4500.),
            'Davis Strait':     (298.5, 66.5, 306.,  67.,  1500.),
            'Hudson Strait':    (295.2, 60.4, 293.7, 61.9, 1000.),
            'Nares Strait':     (284.2, 78.,  287.5, 78.,  1000.),
            'Lancaster Sound':  (281.2, 73.7, 279.7, 74.6, 1000.),
            'Jones Sound':      (279.5, 75.6, 280.,  76.2, 1000.),
            'LabSea Center':    (296.,  63.,  320.,  50.,  4500.),
            }
    if name in info.keys():
        return info.get(name)
    else:
        raise ValueError('Transect \'{:s}\' not found.\n'.format(name) \
                + '- Supported transect names:\n' \
                + '  ' + ', '.join(info.keys()))

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

