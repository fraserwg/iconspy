from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import KDTree
from scipy.stats import mode
import shapely
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyicon as pyic
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

lambert_greenland = ccrs.LambertConformal(central_longitude=-15, standard_parallels=(62, 78))

def create_boundary_connectivity_matrix(ds_IcD, weight_type="distance"):
    i = ds_IcD["edge_vertices"].isel(nv_e=0)
    j = ds_IcD["edge_vertices"].isel(nv_e=1)
    if weight_type == "distance":
        data = ds_IcD["edge_length"] * xr.where(ds_IcD["edge_sea_land_mask"].compute() == 2, 1, np.inf)
    else:
        raise NotImplementedError("The requested weighting type has not yet \
            been implemented")

    iprime = xr.concat((i, j), dim="edge")
    jprime = xr.concat((j, i), dim="edge")
    dataprime = xr.concat((data, data), dim="edge")

    vertex_graph = coo_matrix((dataprime, (iprime.values, jprime.values))).tocsr()
    return vertex_graph


def orientation_along_path(ds_IcD, vertex_path, edge_path):
    # Construct the polygon
    vertex_path = vertex_path.rename(step_in_path_v="step_in_path")
    polygon_xy_pairs = xr.concat(
        (vertex_path["vlon"], vertex_path["vlat"]),
        dim="cart"
    ).transpose("step_in_path", "cart")
    polygon = shapely.Polygon(polygon_xy_pairs)
    
    # Find the adjacent cells
    adj_cell_idx = ds_IcD["adjacent_cell_of_edge"].sel(edge=edge_path).max("nc_e")  # go for max to remove -1 values
    adj_cells = ds_IcD["cell"].sel(cell=adj_cell_idx)

    # Construct an array of shapely points
    adj_cell_xy_pairs = xr.concat((adj_cells["clon"], adj_cells["clat"]), dim="cart").transpose("step_in_path", "cart")
    cell_points = xr.apply_ufunc(
        shapely.Point,
        adj_cell_xy_pairs,
        input_core_dims=[["cart"]],
        vectorize=True,
        # dask="parallelized",
        # dask_gufunc_kwargs={"allow_rechunk": True}
    )

    # Query which points are within the polygon
    cell_points_in_polygon = xr.apply_ufunc(
        polygon.contains,
        cell_points,
        input_core_dims=[[]],
    )
    
    # Create an inside/outside orientation array
    inside_outside_orientation = cell_points_in_polygon.where(cell_points_in_polygon != False, -1)
    
    # Find the orientation of the grid
    ds_IcD2 = ds_IcD.assign_coords(ne_c=("ne_c", [0, 1, 2]))

    edge_of_adj_cells = ds_IcD["edge_of_cell"].sel(cell=adj_cells)

    # Need to find which ne_c index corresponds to our cell/edge pair to get the orientation
    ne_c_index = edge_of_adj_cells.where(edge_of_adj_cells == adj_cells["edge"]).argmax("ne_c")
    grid_orientation = ds_IcD2["orientation_of_normal"].sel(cell=adj_cells).where(ds_IcD2["ne_c"] == ne_c_index).max("ne_c")
    
    path_orientation = inside_outside_orientation * grid_orientation
    path_orientation = path_orientation.drop(["clon", "clat", "cell"])
    return path_orientation


def vertex_path_to_edge_path(ds_IcD, vertex_path):
    """ Converts a path of vertex indices to a path of the edges which
    connect them.

    Paramaters
    ----------
    ds_tgrid : xr.Dataset
        formatted tgrid dataset

    vertex_path : array
        array of vertex indices along a path

    Returns
    -------
    edge_path : array
        array of edge indices connecting the vertices
    """
    vertex_pairs = vertex_path.rolling(step_in_path_v=2).construct(
        window_dim="pair"
    ).isel(step_in_path_v=slice(1, None)).astype("int32")
    
    combined_edges = ds_IcD["edges_of_vertex"].sel(vertex=vertex_pairs).stack(ne_2v=["ne_v", "pair"])

    def _mode(*args, **kwargs):
        vals = mode(*args, **kwargs)
        # only return the mode (discard the count)
        return vals[0].squeeze()

    def mode_xr(obj):
        # note: apply always moves core dimensions to the end
        # usually axis is simply -1 but scipy's mode function doesn't seem to like that
        # this means that this version will only work for DataArray's (not Datasets)
        assert isinstance(obj, xr.DataArray)
        axis = obj.ndim - 1
        return xr.apply_ufunc(_mode, obj,
                              input_core_dims=[["ne_2v"]],
                              kwargs={"axis": axis, "nan_policy": 'omit'},
                              dask="parallelized",
                              dask_gufunc_kwargs={"allow_rechunk": True}
                             )

    edge_path = mode_xr(combined_edges.where(combined_edges != -1)).astype("int32")
    edge_path_xr = ds_IcD["edge"].sel(edge=edge_path)
    edge_path_xr = edge_path_xr.rename(step_in_path_v="step_in_path")
    return edge_path_xr


def find_vertex_path(graph, west_vertex, east_vertex):
    """ Given a graph and the indices of the start and end points, will find
    the shortest path between them

    Parameters
    ----------
    graph : scipy.sparse._csr.csr_matrix
        weighted csr representation of the icon grid

    west_vertex : int
        index of the west vertex of the path
    
    east_vertex : int
        index fo the east vertex of the path

    Returns
    -------
    vertex_path : array
        an array of the indices of the vertices along the shortest path
        from the west to east vertex
    """
    _, predecessors = csgraph.shortest_path(graph, return_predecessors=True, indices=east_vertex)
    path = []
    predecessor = west_vertex
    while True:
        path += [predecessor]
        predecessor = predecessors[predecessor]
        if predecessor == east_vertex:
            path += [predecessor]
            break
    vertex_path = np.array(path)
    return vertex_path

def create_connectivity_matrix(ds_IcD, weights=None):
    i = ds_IcD["edge_vertices"].isel(nv_e=0).astype("int32")
    j = ds_IcD["edge_vertices"].isel(nv_e=1).astype("int32")
    if weights is not None:
        data = abs(weights)
    
    else:
        raise NotImplementedError("The requested weighting type has not yet \
            been implemented")

    iprime = xr.concat((i, j), dim="edge")
    jprime = xr.concat((j, i), dim="edge")
    dataprime = xr.concat((data, data), dim="edge")

    vertex_graph = coo_matrix((dataprime, (iprime.values, jprime.values))).tocsr()
    return vertex_graph


def setup_figure_area(ax=None, proj=None, gridlines=True, coastlines=True, extent=[-85, 30, 55, 85]):
    if proj is None:
        proj = lambert_greenland
    
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": proj}
        )
    else:
        fig = None
    
    if gridlines:
        ax.gridlines()
    if coastlines:
        ax.coastlines()
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def convert_tgrid_data(ds_tgrid, pyic_kwargs=None):
    if pyic_kwargs is None:
        pyic_kwargs = dict()
    
    ds_IcD = pyic.convert_tgrid_data(ds_tgrid, **pyic_kwargs)

    for point in ["cell", "edge", "vertex"]:
        if (point not in ds_IcD.coords) and (point in ds_IcD.dims):
            ds_IcD[point] = np.arange(ds_IcD.dims[point], dtype="int32")

    ds_IcD = ds_IcD.load()
    ds_IcD["edge_vertices"] = ds_IcD["edge_vertices"].astype("int32")
    
    return ds_IcD


##### Pyint functions
class _PyIntKDTree:
    
    def __init__(self, ds_IcD, section_type, scale=1.1):
        self.KDTree = None
        if (section_type == "generic") or (section_type is None):
            self.section_type = "generic"
            self.lon_scale = 1
            self.lat_scale = 1
        elif section_type == "zonal":
            self.section_type = "zonal"
            self.lon_scale = 1
            self.lat_scale = scale
        elif section_type == "meridional":
            self.section_type = "meridional"
            self.lon_scale = scale
            self.lat_scale = 1            
        else:
            raise ValueError(
                f'section_type must be None, "zonal" or "meridional, \
                    not {section_type}"'
                )


class PyIntBoundaryKDTree(_PyIntKDTree):
    def __init__(self, ds_IcD, section_type, scale=1.1):
        super().__init__(ds_IcD, section_type, scale=scale)

        vertices_of_dry_cells = ds_IcD["vertex_of_cell"].where(
            ds_IcD["cell_sea_land_mask"].load() == 1, drop=True
        )
        
        vertices_of_wet_cells = ds_IcD["vertex_of_cell"].where(
            ds_IcD["cell_sea_land_mask"].load() == -1, drop=True
        )
        
        boundary_vertices = np.intersect1d(
            vertices_of_dry_cells, vertices_of_wet_cells
        ).astype("int32")

        self.boundary_vertex_pairs = xr.concat(
            (
                ds_IcD["vlon"].sel(vertex=boundary_vertices) * self.lon_scale,
                ds_IcD["vlat"].sel(vertex=boundary_vertices) * self.lat_scale,
            ),
            dim="cart_h",
        ).transpose(..., "cart_h")
        
        # Construct a kdTree from these vertices
        self.KDTree = KDTree(self.boundary_vertex_pairs)


def find_boundary_vertex(ds_IcD,
                         lon=None,
                         lat=None,
                         boundary_KDTree=None,
                         section_type=None,
                         query_kwargs=None):
    """finds vertex on boundary near lat and lon

    Args:
        lat (float): approximate latitude
        lon (float): approximate longitude
        section_type (str | None): type of section. Can be either "zonal",
            "meridional" or None
    """
    if (lon is None) or (lat is None):
        raise ValueError('variables "lon" and "lat" must be specified. They \
            are kwargs purely to avoid problems arising from mixing them up.')
    
    # Format lon and lat as arrays
    if np.isscalar(lon):
        lon = [lon]
    if np.isscalar(lat):
        lat = [lat]
    lon, lat = np.asarray(lon), np.asarray(lat)
    
    if boundary_KDTree is not None:
        assert isinstance(boundary_KDTree, PyIntBoundaryKDTree)
        if section_type is not None:
            assert boundary_KDTree.section_type == section_type
    else:
        boundary_KDTree = PyIntBoundaryKDTree(ds_IcD, section_type)
    
    if query_kwargs is None:
        query_kwargs = dict()
    
    # Perform the query
    slon = lon * boundary_KDTree.lon_scale
    slat = lat * boundary_KDTree.lat_scale
    query_points = list(zip(slon, slat))
    _, vidx = boundary_KDTree.KDTree.query(query_points, **query_kwargs)
    
    return boundary_KDTree.boundary_vertex_pairs["vertex"].isel(vertex=vidx)






class PyIntWetKDTree(_PyIntKDTree):
    def __init__(self, ds_IcD, section_type, scale=1.1):
        super().__init__(ds_IcD, section_type, scale=scale)        
        
        vertex_pairs = xr.concat(
            (ds_IcD["vlon"], ds_IcD["vlat"]),
            dim="cart_h"
        ).transpose(..., "cart_h")
        
        self.KDTree = KDTree(vertex_pairs)


def find_wet_vertex(ds_IcD, 
                    lon=None,
                    lat=None,
                    wet_KDTree=None,
                    section_type=None,
                    query_kwargs=None,
                    assert_wet=True):
    
    if (lon is None) or (lat is None):
        raise ValueError('variables "lon" and "lat" must be specified. They \
            are kwargs purely to avoid problems arising from mixing them up.')

    # Format lon and lat as arrays
    if np.isscalar(lon):
        lon = [lon]
    if np.isscalar(lat):
        lat = [lat]
    lon, lat = np.asarray(lon), np.asarray(lat)
        
    if wet_KDTree is not None:
        assert isinstance(wet_KDTree, PyIntWetKDTree)
        if section_type is not None:
            assert wet_KDTree.section_type == section_type
    else:
        wet_KDTree = PyIntWetKDTree(ds_IcD, section_type)

    # Construct a kdTree from the IcD vertices
    vertex_pairs = xr.concat(
        (ds_IcD["vlon"], ds_IcD["vlat"]),
        dim="cart_h"
    ).transpose(..., "cart_h")

    if query_kwargs is None:
        query_kwargs = dict()

    # Perform the query
    slon = lon * wet_KDTree.lon_scale
    slat = lat * wet_KDTree.lat_scale
    query_points = list(zip(slon, slat))
    _, vidx = wet_KDTree.KDTree.query(query_points, **query_kwargs)

    # Verify the point is wet if requested
    if assert_wet:
        if not _is_vertex_wet(ds_IcD, vidx):
            raise RuntimeError("The vertex found was either a boundary or \
                land point. Try choosing a wetter start position, or run with \
                'assert_wet=False'.")

    return ds_IcD["vertex"].isel(vertex=vidx)


def _is_vertex_wet(ds_IcD, vidx):
    adjacent_cells = ds_IcD["cells_of_vertex"].isel(vertex=vidx).load()
    adjacent_cell_mask = ds_IcD["cell_sea_land_mask"].sel(cell=adjacent_cells)
    
    if (1 in adjacent_cell_mask) or (2 in adjacent_cell_mask):
        return False
    else:
        return True