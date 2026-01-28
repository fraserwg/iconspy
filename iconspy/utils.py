from itertools import product
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import KDTree
from scipy.stats import mode
import shapely
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

lambert_greenland = ccrs.LambertConformal(central_longitude=-15, standard_parallels=(62, 78))

def create_boundary_connectivity_matrix(ds_IsD, weight_type="distance"):
    i = ds_IsD["edge_vertices"].isel(nv_e=0)
    j = ds_IsD["edge_vertices"].isel(nv_e=1)
    if weight_type == "distance":
        data = ds_IsD["edge_length"] * xr.where(ds_IsD["edge_sea_land_mask"].compute() == 2, 1, np.inf)
    else:
        raise NotImplementedError("The requested weighting type has not yet \
            been implemented")

    iprime = xr.concat((i, j), dim="edge")
    jprime = xr.concat((j, i), dim="edge")
    dataprime = xr.concat((data, data), dim="edge")

    vertex_graph = coo_matrix((dataprime, (iprime.values, jprime.values))).tocsr()
    return vertex_graph


def orientation_along_path(ds_IsD, vertex_path, edge_path):
    # Construct the polygon
    vertex_path = vertex_path.rename(step_in_path_v="step_in_path")
    polygon_xy_pairs = xr.concat(
        (vertex_path["vlon"], vertex_path["vlat"]),
        dim="cart"
    ).transpose("step_in_path", "cart")
    polygon = shapely.Polygon(polygon_xy_pairs)
    
    # Find the adjacent cells
    adj_cell_idx = ds_IsD["adjacent_cell_of_edge"].sel(edge=edge_path).max("nc_e")  # go for max to remove -1 values
    adj_cells = ds_IsD["cell"].sel(cell=adj_cell_idx)

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
    ds_IsD2 = ds_IsD.assign_coords(ne_c=("ne_c", [0, 1, 2]))

    edge_of_adj_cells = ds_IsD["edge_of_cell"].sel(cell=adj_cells)

    # Need to find which ne_c index corresponds to our cell/edge pair to get the orientation
    ne_c_index = edge_of_adj_cells.where(edge_of_adj_cells == adj_cells["edge"]).argmax("ne_c")
    grid_orientation = ds_IsD2["orientation_of_normal"].sel(cell=adj_cells).where(ds_IsD2["ne_c"] == ne_c_index).max("ne_c")
    
    path_orientation = inside_outside_orientation * grid_orientation
    path_orientation = path_orientation.drop(["clon", "clat", "cell"])
    return path_orientation


def vertex_path_to_edge_path(ds_IsD, vertex_path):
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
    
    combined_edges = ds_IsD["edges_of_vertex"].sel(vertex=vertex_pairs).stack(ne_2v=["ne_v", "pair"])

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
    edge_path_xr = ds_IsD["edge"].sel(edge=edge_path)
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

def create_connectivity_matrix(ds_IsD, weights=None):
    i = ds_IsD["edge_vertices"].isel(nv_e=0).astype("int32")
    j = ds_IsD["edge_vertices"].isel(nv_e=1).astype("int32")
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


def setup_figure_area(ax=None, proj=None, gridlines=True, coastlines=True, extent=None):
    if proj is None:
        # proj = lambert_greenland
        proj = ccrs.PlateCarree()
    
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


def _pyicon_convert_tgrid_data(ds_tg_in):
    """Convert xarray grid file to grid file compatible with pyicon function.

    Parameters
    ----------
    ds_tg_in : xr.Dataset
        raw, unprocessed tgrid

    Returns
    -------
    ds_IcD : xr.Dataset
        A tgrid dataset compatible with pyicon functions


    Notes
    -----
    Open classical ICON grid file by:
    >>> ds_tg = xr.open_dataset(fpath_tg, chunks=dict())

    Then convert by:
    >>> ds_IcD = pyic.convert_tgrid_data(ds_tg)
    
    Notes
    -----
    Original code from pyicon under the MIT license.
    Modified by Fraser Goldsworth on 18.06.2025.
    See LICENSE file for more information.

    """

    # make deep copy of ds_tg_in to avoid glaobal modifications if during this function call
    ds_tg = ds_tg_in.copy(deep=True)

    if "converted_tgrid" in ds_tg.attrs:
        raise ValueError(
            "ds_tg has previously been converted by this function," + \
            "applying the function again will lead to undocumented" + \
            "behaviour."
        )

    ds_IcD = xr.Dataset()

    # --- constants (from src/shared/mo_physical_constants.f90)
    ds_IcD["grid_sphere_radius"] = 6.371229e6
    ds_IcD["grav"] = 9.80665
    ds_IcD["earth_angular_velocity"] = 7.29212e-05
    ds_IcD["rho0"] = 1025.022
    ds_IcD["rhoi"] = 917.0
    ds_IcD["rhos"] = 300.0
    ds_IcD["sal_ref"] = 35.0
    ds_IcD["sal_ice"] = 5.0
    rcpl = 3.1733
    cpd = 1004.64
    ds_IcD["cp"] = (rcpl + 1.0) * cpd
    ds_IcD["tref"] = 273.15
    ds_IcD["tmelt"] = 273.15
    ds_IcD["tfreeze"] = -1.9
    ds_IcD["alf"] = 2.8345e6 - 2.5008e6  # [J/kg]   latent heat for fusion

    # --- distances and areas
    ds_IcD["cell_area"] = ds_tg["cell_area"]
    ds_IcD["cell_area_p"] = ds_tg["cell_area_p"]
    ds_IcD["dual_area"] = ds_tg["dual_area"]
    ds_IcD["edge_length"] = ds_tg["edge_length"]
    ds_IcD["dual_edge_length"] = ds_tg["dual_edge_length"]
    ds_IcD["edge_cell_distance"] = ds_tg["edge_cell_distance"].transpose()
    # --- neighbor information
    ds_IcD["vertex_of_cell"] = ds_tg["vertex_of_cell"].transpose() - 1
    ds_IcD["edge_of_cell"] = ds_tg["edge_of_cell"].transpose() - 1
    ds_IcD["vertices_of_vertex"] = ds_tg["vertices_of_vertex"].transpose() - 1
    ds_IcD["edges_of_vertex"] = ds_tg["edges_of_vertex"].transpose() - 1
    ds_IcD["edge_vertices"] = ds_tg["edge_vertices"].transpose() - 1
    ds_IcD["adjacent_cell_of_edge"] = ds_tg["adjacent_cell_of_edge"].transpose() - 1
    ds_IcD["cells_of_vertex"] = ds_tg["cells_of_vertex"].transpose() - 1
    ds_IcD["adjacent_cell_of_cell"] = ds_tg["neighbor_cell_index"].transpose() - 1
    # --- orientation
    ds_IcD["orientation_of_normal"] = ds_tg["orientation_of_normal"].transpose()
    ds_IcD["edge_orientation"] = ds_tg["edge_orientation"].transpose()
    ds_IcD["tangent_orientation"] = ds_tg["edge_system_orientation"].transpose()

    # --- masks
    ds_IcD["cell_sea_land_mask"] = ds_tg["cell_sea_land_mask"]
    ds_IcD["edge_sea_land_mask"] = ds_tg["edge_sea_land_mask"]

    # --- coordinates
    ds_IcD["cell_cart_vec"] = xr.concat(
        [
            ds_tg["cell_circumcenter_cartesian_x"],
            ds_tg["cell_circumcenter_cartesian_y"],
            ds_tg["cell_circumcenter_cartesian_z"],
        ],
        dim="cart",
    ).transpose()

    ds_IcD["vert_cart_vec"] = xr.concat(
        [
            ds_tg["cartesian_x_vertices"],
            ds_tg["cartesian_y_vertices"],
            ds_tg["cartesian_z_vertices"],
        ],
        dim="cart",
    ).transpose()

    ds_IcD["edge_cart_vec"] = xr.concat(
        [
            ds_tg["edge_middle_cartesian_x"],
            ds_tg["edge_middle_cartesian_y"],
            ds_tg["edge_middle_cartesian_z"],
        ],
        dim="cart",
    ).transpose()

    ds_IcD["dual_edge_cart_vec"] = xr.concat(
        [
            ds_tg["edge_dual_middle_cartesian_x"],
            ds_tg["edge_dual_middle_cartesian_y"],
            ds_tg["edge_dual_middle_cartesian_z"],
        ],
        dim="cart",
    ).transpose()

    ds_IcD["edge_prim_norm"] = xr.concat(
        [
            ds_tg["edge_primal_normal_cartesian_x"],
            ds_tg["edge_primal_normal_cartesian_y"],
            ds_tg["edge_primal_normal_cartesian_z"],
        ],
        dim="cart",
    ).transpose()

    for point, dim in product("ecv", ("lat", "lon")):
        coord = point + dim
        ds_IcD[coord] *= 180.0 / np.pi
        ds_IcD[coord].attrs["units"] = "degrees"

    ds_IcD["fc"] = (
        2.0 * ds_IcD.earth_angular_velocity * np.sin(ds_IcD.clat * np.pi / 180.0)
    )
    ds_IcD["fe"] = (
        2.0 * ds_IcD.earth_angular_velocity * np.sin(ds_IcD.elat * np.pi / 180.0)
    )
    ds_IcD["fv"] = (
        2.0 * ds_IcD.earth_angular_velocity * np.sin(ds_IcD.vlat * np.pi / 180.0)
    )

    try:
        ds_IcD = ds_IcD.rename({"ncells": "cell"})
    except ValueError:
        pass

    # Default dimension names are messy and often wrong. Let's rename them.
    dim_name_remappings = {
        "vertex_of_cell": {"nv": "nv_c"},
        "edge_vertices": {"nc": "nv_e"},
        "vertices_of_vertex": {"ne": "nv_v"},
        "edge_of_cell": {"nv": "ne_c"},
        "edges_of_vertex": {"ne": "ne_v"},
        "adjacent_cell_of_edge": {"nc": "nc_e"},
        "cells_of_vertex": {"ne": "nc_v"},
        "edge_cell_distance": {"nc": "nc_e"},
        "orientation_of_normal": {"nv": "ne_c"},
        "edge_orientation": {"ne": "ne_v"},
        "adjacent_cell_of_cell": {"nv": "nc_c"},
    }

    for variable in dim_name_remappings:
        ds_IcD[variable] = ds_IcD[variable].rename(dim_name_remappings[variable])

    ds_IcD.attrs["converted_tgrid"] = True
    ds_tg.attrs["converted_tgrid"] = True

    standard_order = ["cell", "vertex", "edge", "nc", "nv", "ne", "cart", ...]
    ds_IcD = ds_IcD.transpose(*standard_order, missing_dims="ignore")

    return ds_IcD



def convert_tgrid_data(ds_tgrid, pyic_kwargs=None):
    """Formats the model grid in the format required by iconspy

    Parameters
    ----------
    ds_tgrid : xarray.Dataset
        Dataset represention of the raw model grid (e.g. downloaded from "http://icon-downloads.mpimet.mpg.de")
    pyic_kwargs : dict, optional
        Dictionary containig arguments to be passed to the pyicon.convert_tgrid_data function, by default None

    Returns
    -------
    ds_IsD : xarray.Dataset
        Dataset represention of the model grid in the format required by iconspy
        
    Notes
    -----
    An iconspy dataset (ds_IsD) is similar to but distinct from a pyicon dataset (ds_IcD).
    I suggest loading the dataset having loaded it
    
    Example
    -------
    >>> from pathlib import Path
    >>> import xarray as xr
    >>> import iconspy as ispy
    >>> tgrid_path = Path("/pool/data/ICON/grids/public/mpim/0036/icon_grid_0036_R02B04_O.nc")
    >>> ds_tgrid = xr.open_dataset(tgrid_path)
    >>> ds_IsD = convert_tgrid_data(ds_tgrid)
    >>> ds_IsD = ds_IsD.load()
    """
    if "IsD_compatible_flag" in ds_tgrid.attrs:
        if ds_tgrid.attrs["IsD_compatible_flag"] == True:
            raise ValueError(
                "ds_tgrid has previously been converted by this function," + \
                "applying the function again will lead to undocumented" + \
                "behaviour."
            )
    
    if pyic_kwargs is None:
        pyic_kwargs = dict()
    
    ds_IsD = _pyicon_convert_tgrid_data(ds_tgrid, **pyic_kwargs)

    for point in ["cell", "edge", "vertex"]:
        if (point not in ds_IsD.coords) and (point in ds_IsD.dims):
            ds_IsD[point] = np.arange(ds_IsD.sizes[point], dtype="int32")

    ds_IsD = ds_IsD.load()
    ds_IsD["edge_vertices"] = ds_IsD["edge_vertices"].astype("int32")
    
    ds_IsD.attrs["IsD_compatible_flag"] = True
    
    return ds_IsD

def _assert_IsD_compatible(ds_IsD):
    """ Checks whether a dataset is compatible with iconspy functions

    Parameters
    ----------
    ds_IsD : xarray.Dataset
        Dataset to be checked
    """
    # No flag
    if ("IsD_compatible_flag" not in ds_IsD.attrs):
        raise ValueError("The ds_IsD dataset provided is not an iconspy compatible dataset. Have you ispy.convert_tgrid_data() on it to format it correctly?")
    
    # Flag is incorrect
    elif ds_IsD.attrs["IsD_compatible_flag"] != True:
        raise ValueError("The ds_IsD dataset provided is explicitly not an iconspy compatible dataset.")
