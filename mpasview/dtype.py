#--------------------------------
# Unstractured data type
#--------------------------------

import numpy as np
import copy

#--------------------------------
# Named numpy array
#--------------------------------
class NNArray(np.ndarray):

    """Named numpy array

    """

    def __new__(cls, input_array, name='', units=''):
        obj = np.asarray(input_array).view(cls)
        obj.name = str(name)
        obj.units = str(units)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', '')
        self.units = getattr(obj, 'units', '')

#--------------------------------
# UArray
#--------------------------------

class UArray2D(object):

    """Simple two-dimensional unstructured data array

    """

    def __init__(
            self,
            data = np.nan,
            name = '',
            units = '',
            x = np.nan,
            xname = '',
            xunits = '',
            y = np.nan,
            yname = '',
            yunits ='',
            ):
        """Initialization of UArray2D

        :data:      (list or numpy array) data array
        :name:      (str) name of data
        :units:     (str) units of data
        :x:         (list or numpy array) x-dimension
        :xname:     (str) name of x
        :xunits:    (str) units of x
        :y:         (list or numpy array) y-dimension
        :yname:     (str) name of y
        :yunits:    (str) units of y

        """
        # check input
        assert np.asarray(data).shape == np.asarray(x).shape, 'Shape of x mismatches data.'
        assert np.asarray(data).shape == np.asarray(y).shape, 'Shape of y mismatches data.'

        # initialize UArray2D
        self.data = NNArray(data, name=name, units=units).flatten()
        self.x = NNArray(x, name=xname, units=xunits).flatten()
        self.y = NNArray(y, name=yname, units=yunits).flatten()
        self.size = np.asarray(data).size

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+' (size={}):'.format(self.size)]
        for attr in ['data', 'x', 'y']:
            darr = getattr(self,attr)
            if self.size > 4:
                dataview = '[{:f} {:f} ... {:f} {:f}]'.format(darr[0], darr[1], darr[-2], darr[-1])
            else:
                dataview = str(darr)
            summary.append('{:>8s}: '.format(attr)+dataview)
            summary.append('{:>8s}: '.format('name')+getattr(getattr(self, attr),'name'))
            summary.append('{:>8s}: '.format('units')+getattr(getattr(self, attr),'units'))
            summary.append('')
        return '\n'.join(summary)

    def __getitem__(self, index):
        """Slicing

        """
        out = copy.copy(self)
        for attr in ['data', 'x', 'y']:
            setattr(out, attr, getattr(self, attr)[index])
        out.size = out.data.size
        return out

#--------------------------------
# UMesh
#--------------------------------

class UMesh(object):

    """Mesh information for unstructured data array"""

    def __init__(self):
        """Initialization of UMesh

        :name:          (str) name of mesh
        :ncells:        (int) number of cells
        :cellid:        (array of shape ncells) cell IDs
        :xcell:         (array of shape ncells) x-coordinate of cells
        :ycell:         (array of shape ncells) y-coordinate of cells
        :acell:         (array of shape ncells) area of cells
        :nedges_cell:   (array of shape ncells) number of edges on cells
        :edges_cell:    (array of shape ncells times maxedges) edges on cells
        :nedges:        (int) number of edges
        :edgeid:        (array of nedges) edge IDs
        :xedge:         (array of nedges) x-coordinate of edges
        :yedge:         (array of nedges) y-coordinate of edges
        :dc_edge:       (array of nedges) length of edges, distance between cells on edge
        :dv_edge:       (array of nedges) length of edges, distance between vertices on edge
        :cells_edge:    (array of nedges times 2) cells on edges
        :vertices_edge: (array of nedges times 2) verticies on edges
        :nverticies:    (int) number of verticies
        :vertexid:      (array of nvertices) vertex IDs
        :xvertex:       (array of nvertices) x-coordinate of vertices
        :yvertex:       (array of nvertices) y-coordinate of vertices
        :adual:         (array of nvertices) area of dual cells on vertices
        :cells_vertex:  (array of nvertices times vertexdegree) cells on vertices
        :edges_vertex:  (array of nvertices times vertexdegree) edges on vertices

        """
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

    def __repr__(self):
        """Formatted print

        """
        summary = [str(self.__class__)+':']
        summary.append('{:>10s}: {:s}'.format('name', self.name))
        for attr in ['ncells', 'nedges', 'nvertices']:
            summary.append('{:>10s}: {:d}'.format(attr, getattr(self, attr)))
        return '\n'.join(summary)

