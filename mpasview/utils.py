#--------------------------------
# Functions
#--------------------------------

import numpy as np
from scipy import spatial
from mpl_toolkits.basemap import Basemap

#--------------------------------
# mesh
#--------------------------------

class Path:

    """Path object

    """

    def __init__(
            self,
            xvertex = [],
            yvertex = [],
            xedge = [],
            yedge = [],
            sign_edges = [],
            ):
        """Initialize Path object

        :xvertex:     (array-like, optional) x-coordinate of vertices
        :yvertex:     (array-like, optional) y-coordinate of vertices
        :xedge:       (array-like, optional) x-coordinate of edges
        :yedge:       (array-like, optional) y-coordinate of edges
        :sign_edges:  (array-like, optional) sign of edges

        """

        self.xvertex = list(xvertex)
        self.yvertex = list(yvertex)
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
        if isinstance(axis, Basemap):
            xx, yy = axis(self.xedge, self.yedge)
            out = axis.scatter(xx, yy, s=s, **kwargs)
        else:
            out = axis.scatter(self.xedge, self.yedge, s=s, **kwargs)
        return out

    def project_vertex(self, axis, s=1, **kwargs):
        if isinstance(axis, Basemap):
            xx, yy = axis(self.xvertex, self.yvertex)
            out = axis.scatter(xx, yy, s=s, **kwargs)
        else:
            out = axis.scatter(self.xvertex, self.yvertex, s=s, **kwargs)
        return out

    def project_edge(self, axis, **kwargs):
        if isinstance(axis, Basemap):
            xx, yy = axis(self.xvertex, self.yvertex)
            out = axis.plot(xx, yy, **kwargs)
        else:
            out = axis.plot(self.xvertex, self.yvertex, **kwargs)
        return out

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

def get_path(
        vidx_p0,
        vidx_p1,
        xvertex,
        yvertex,
        xedge,
        yedge,
        vertexid,
        edges_vertex,
        vertices_edge,
        on_sphere = True,
        debug_info = False,
        ):
    """Get the path between two endpoints p0 and p1

    :vidx_p0:       (int) vertex index of p0
    :vidx_p1:       (int) vertex index of p1
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :xedge:         (array-like) x-coordinate of edges
    :yedge:         (array-like) y-coordinate of edges
    :vertexid:      (array-like) vertex ID
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
    # record vortices on path and the indices
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
    x_edge = xedge[idx_edges_on_path]
    y_edge = yedge[idx_edges_on_path]
    x_vertex = xvertex[idx_vertices_on_path]
    y_vertex = yvertex[idx_vertices_on_path]
    out = Path(
            xvertex=x_vertex,
            yvertex=y_vertex,
            xedge=x_edge,
            yedge=y_edge,
            sign_edges=sign_edges,
            )
    return out

#--------------------------------
# utility
#--------------------------------

def get_region_llrange(region):
    """Get the range of longitude and latitude in a region.

    :region:    (str) region name
    :returns:   (tuple) (lon_min, lon_max, lat_min, lat_max)

    """
    llrange = {
            'Global': (  0., 360., -90., 90.),
            'Arctic': (  0., 360.,  45., 90.),
            'LabSea': (270., 356.,  36., 75.),
            'TropicalPacific':  (130., 290., -20., 20.),
            'TropicalAtlantic': (310., 380., -20., 20.),
            }
    if region in llrange.keys():
        return llrange.get(region)
    else:
        raise ValueError('Region \'{:s}\' not found.\n'.format(region) \
                + '- Supported region names:\n' \
                + '  ' + ', '.join(switcher.keys()))

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
    for i in idx[0][:]:
        if y_arr[i] == y_sub[cidx]:
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

