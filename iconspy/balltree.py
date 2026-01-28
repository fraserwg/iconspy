import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree


class _IspyBallTree:    
    def __init__(self, ds_IsD):
        pass
        
class IspyBoundaryBallTree(_IspyBallTree):
    def __init__(self, ds_IsD):
        super().__init__(ds_IsD)

        vertices_of_dry_cells = ds_IsD["vertex_of_cell"].where(
            ds_IsD["cell_sea_land_mask"].load() == 1, drop=True
        )
        
        vertices_of_wet_cells = ds_IsD["vertex_of_cell"].where(
            ds_IsD["cell_sea_land_mask"].load() == -1, drop=True
        )
        
        boundary_vertices = np.intersect1d(
            vertices_of_dry_cells, vertices_of_wet_cells
        ).astype("int32")

        self.boundary_vertex_pairs = xr.concat(
            (
                np.radians(ds_IsD["vlat"].sel(vertex=boundary_vertices)),
                np.radians(ds_IsD["vlon"].sel(vertex=boundary_vertices)),
            ),
            dim="cart_h",
        ).transpose(..., "cart_h")
        
        # Construct a BallTree from these vertices
        self.BallTree = BallTree(self.boundary_vertex_pairs, metric='haversine')


def find_boundary_vertex(ds_IsD,
                         lon=None,
                         lat=None,
                         boundary_BallTree=None,
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
    lon, lat = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
    
    
    
    if boundary_BallTree is not None:
        assert isinstance(boundary_BallTree, IspyBoundaryBallTree)
    else:
        boundary_BallTree = IspyBoundaryBallTree(ds_IsD)
    
    if query_kwargs is None:
        query_kwargs = dict()
    
    # Perform the query
    query_points = list(zip(lat, lon))
    _, vidx = boundary_BallTree.BallTree.query(query_points, **query_kwargs)
    
    return boundary_BallTree.boundary_vertex_pairs["vertex"].isel(vertex=vidx)


class IspyWetBallTree(_IspyBallTree):
    def __init__(self, ds_IsD):
        super().__init__(ds_IsD)        
        
        self.vertex_pairs = xr.concat(
            (
                np.radians(ds_IsD["vlat"]),
                np.radians(ds_IsD["vlon"])),
            dim="cart_h"
        ).transpose(..., "cart_h")
        
        self.BallTree = BallTree(self.vertex_pairs, metric="haversine")


def find_wet_vertex(ds_IsD, 
                    lon=None,
                    lat=None,
                    wet_BallTree=None,
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
    lon, lat = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
        
    if wet_BallTree is not None:
        assert isinstance(wet_BallTree, IspyWetBallTree)
    else:
        wet_BallTree = IspyWetBallTree(ds_IsD)

    if query_kwargs is None:
        query_kwargs = dict()

    # Perform the query
    query_points = list(zip(lat, lon))
    _, vidx = wet_BallTree.BallTree.query(query_points, **query_kwargs)
    vidx = vidx.squeeze()
    
    # Verify the point is wet if requested
    if assert_wet:
        if not _is_vertex_wet(ds_IsD, vidx):
                raise RuntimeError(f"The vertex {vidx} was either a boundary or \
                land point. Try choosing a wetter start position, or run with \
                'assert_wet=False'.")

    # return vidx
    return ds_IsD["vertex"].isel(vertex=vidx.squeeze())


def _is_vertex_wet(ds_IsD, vidx):
    adjacent_cells = ds_IsD["cells_of_vertex"].isel(vertex=vidx).load()
    adjacent_cell_mask = ds_IsD["cell_sea_land_mask"].isel(cell=adjacent_cells)
    
    if (1 in adjacent_cell_mask) or (2 in adjacent_cell_mask):
        return False
    else:
        return True