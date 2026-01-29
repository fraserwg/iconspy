from iconspy.balltree import *
from iconspy.balltree import _is_vertex_wet
from .conftest import raw_grid, ispy_grid, boundary_target_station
import numpy as np


def test_IspyBoundaryBallTree(ispy_grid):
    ds_IsD = ispy_grid
    boundary_BallTree = IspyBoundaryBallTree(ds_IsD)

    lat_lon_idx = [
        # Random points
        [3.81, -49.84, 2749],
        [-20, 0, 4325],
        
        # Point near the dateline
        [0, 180, 5181],
        [0, -180, 5181],
        
        # Check positive and negative longitudes work
        [0, 270, 4661],
        [0, -90, 4661],
        
        # North Pole
        [90, 0, 977],
        [90, 160, 977],
    ]

    for target_lat, target_lon, expected_vertex in lat_lon_idx:
        _, _idx = boundary_BallTree.BallTree.query([[np.radians(target_lat), np.radians(target_lon)]])
        _idx = _idx.squeeze()
        idx = boundary_BallTree.boundary_vertex_pairs["vertex"].isel(vertex=_idx)
        idx = idx.squeeze()
        assert idx == expected_vertex


def test_find_boundary_vertex(ispy_grid):
    ds_IsD = ispy_grid

    lat_lon_idx = [
        # Random points
        [3.81, -49.84, 2749],
        [-20, 0, 4325],
        
        # Point near the dateline
        [0, 180, 5181],
        [0, -180, 5181],
        
        # Check positive and negative longitudes work
        [0, 270, 4661],
        [0, -90, 4661],
        
        # North Pole
        [90, 0, 977],
        [90, 160, 977],
    ]

    for target_lat, target_lon, expected_vertex in lat_lon_idx:
        vidx = find_boundary_vertex(
            ds_IsD,
            lon=target_lon,
            lat=target_lat,
        )
        vidx = vidx.squeeze().values

        assert vidx == expected_vertex


def test_IspyWetBallTree(ispy_grid):
    ds_IsD = ispy_grid
    wet_BallTree = IspyWetBallTree(ds_IsD)

    lat_lon_idx = [
        # Random points
        [0, -40, 3537],
        [-20, 0, 4460],
        
        # Point near the dateline
        [0, 180, 3221],
        [0, -180, 3221],
        
        # Check positive and negative longitudes work
        [0, 270, 3367],
        [0, -90, 3367],
        
        # North Pole
        [90, 0, 57],
        [90, 160, 57],
    ]    

    for target_lat, target_lon, expected_vertex in lat_lon_idx:
        _, idx = wet_BallTree.BallTree.query([[np.radians(target_lat), np.radians(target_lon)]])
        idx = idx.squeeze()
        assert idx == expected_vertex

def test_find_wet_vertex(ispy_grid):
    ds_IsD = ispy_grid

    lat_lon_idx = [
        # Random points
        [0, -40, 3537],
        [-20, 0, 4460],
        
        # Point near the dateline
        [0, 180, 3221],
        [0, -180, 3221],
        
        # Check positive and negative longitudes work
        [0, 270, 3367],
        [0, -90, 3367],
        
        # North Pole
        [90, 0, 57],
        [90, 160, 57],
    ]    
    
    for target_lat, target_lon, expected_vertex in lat_lon_idx:
        vidx = find_wet_vertex(
            ds_IsD,
            lon=target_lon,
            lat=target_lat,
            assert_wet=True,
        )
        vidx = vidx.squeeze().values

        assert vidx == expected_vertex