import numpy as np
import xarray as xr
from scipy.spatial import KDTree


class _IspyKDTree:
    
    def __init__(self, ds_IsD, section_type, scale=1.1):
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


class IspyBoundaryKDTree(_IspyKDTree):
    def __init__(self, ds_IsD, section_type, scale=1.1):
        super().__init__(ds_IsD, section_type, scale=scale)

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
                ds_IsD["vlon"].sel(vertex=boundary_vertices) * self.lon_scale,
                ds_IsD["vlat"].sel(vertex=boundary_vertices) * self.lat_scale,
            ),
            dim="cart_h",
        ).transpose(..., "cart_h")
        
        # Construct a kdTree from these vertices
        self.KDTree = KDTree(self.boundary_vertex_pairs)


def find_boundary_vertex(ds_IsD,
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
        assert isinstance(boundary_KDTree, IspyBoundaryKDTree)
        if section_type is not None:
            assert boundary_KDTree.section_type == section_type
    else:
        boundary_KDTree = IspyBoundaryKDTree(ds_IsD, section_type)
    
    if query_kwargs is None:
        query_kwargs = dict()
    
    # Perform the query
    slon = lon * boundary_KDTree.lon_scale
    slat = lat * boundary_KDTree.lat_scale
    query_points = list(zip(slon, slat))
    _, vidx = boundary_KDTree.KDTree.query(query_points, **query_kwargs)
    
    return boundary_KDTree.boundary_vertex_pairs["vertex"].isel(vertex=vidx)






class IspyWetKDTree(_IspyKDTree):
    def __init__(self, ds_IsD, section_type, scale=1.1):
        super().__init__(ds_IsD, section_type, scale=scale)        
        
        vertex_pairs = xr.concat(
            (ds_IsD["vlon"], ds_IsD["vlat"]),
            dim="cart_h"
        ).transpose(..., "cart_h")
        
        self.KDTree = KDTree(vertex_pairs)


def find_wet_vertex(ds_IsD, 
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
        assert isinstance(wet_KDTree, IspyWetKDTree)
        if section_type is not None:
            assert wet_KDTree.section_type == section_type
    else:
        wet_KDTree = IspyWetKDTree(ds_IsD, section_type)

    # Construct a kdTree from the IcD vertices
    vertex_pairs = xr.concat(
        (ds_IsD["vlon"], ds_IsD["vlat"]),
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
        if not _is_vertex_wet(ds_IsD, vidx):
            raise RuntimeError("The vertex found was either a boundary or \
                land point. Try choosing a wetter start position, or run with \
                'assert_wet=False'.")

    return ds_IsD["vertex"].isel(vertex=vidx)


def _is_vertex_wet(ds_IsD, vidx):
    adjacent_cells = ds_IsD["cells_of_vertex"].isel(vertex=vidx).load()
    adjacent_cell_mask = ds_IsD["cell_sea_land_mask"].sel(cell=adjacent_cells)
    
    if (1 in adjacent_cell_mask) or (2 in adjacent_cell_mask):
        return False
    else:
        return True