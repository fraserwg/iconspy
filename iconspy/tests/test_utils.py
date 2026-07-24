import pytest
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix
from iconspy.utils import (
    convert_tgrid_data,
    _assert_IsD_compatible,
    create_connectivity_matrix,
    find_vertex_path,
    vertex_path_to_edge_path,
    orientation_along_path,
)
from iconspy.core import Section, Region, TargetStation
from iconspy.balltree import IspyBoundaryBallTree, IspyWetBallTree
from .conftest import raw_grid, ispy_grid, boundary_target_station


def test_IsD_conversion(raw_grid):
    ds_tgrid = raw_grid
    
    # Check conversion to IsD works
    ds_IsD = convert_tgrid_data(ds_tgrid)
    
    # Check required attributes are present and functioning
    assert isinstance(ds_IsD, xr.Dataset)
    assert ds_IsD.attrs["uuidOfHGrid"] == "5bd948e8-ac1a-11ea-a6b1-d317264fdca9"
    assert isinstance(ds_IsD.attrs["boundary_BallTree"], IspyBoundaryBallTree)
    assert isinstance(ds_IsD.attrs["wet_BallTree"], IspyWetBallTree)
    
    # Check compatibility assertion is working
    _assert_IsD_compatible(ds_IsD)
    
    # Check that conversion of a converted datset fails.
    with pytest.raises(ValueError):
        _ = convert_tgrid_data(ds_IsD)
    
    # Check tgrid is not compatible
    with pytest.raises(ValueError):
        _assert_IsD_compatible(ds_tgrid)
    
    # Check that if the flag is present but not True we break things
    ds_IsD_wrong_flag = ds_IsD.copy().assign_attrs(IsD_compatible_flag=False)
    with pytest.raises(ValueError):
        _assert_IsD_compatible(ds_IsD_wrong_flag)


def test_create_connectivity_matrix(ispy_grid):
    ds_IsD = ispy_grid
    n_vertices = ds_IsD.sizes["vertex"]

    # With edge_length weights
    graph = create_connectivity_matrix(ds_IsD, ds_IsD["edge_length"])
    assert isinstance(graph, csr_matrix)
    assert graph.shape == (n_vertices, n_vertices)
    assert graph.nnz == 46414
    assert (graph.data > 0).all()

    # With uniform weights all non-zero entries should equal 1
    uniform_weights = xr.ones_like(ds_IsD["edge_length"])
    graph_uniform = create_connectivity_matrix(ds_IsD, uniform_weights)
    assert graph_uniform.shape == graph.shape
    assert graph_uniform.nnz == graph.nnz
    assert np.allclose(graph_uniform.data, 1.0)


def test_find_vertex_path(ispy_grid):
    ds_IsD = ispy_grid
    graph = create_connectivity_matrix(ds_IsD, ds_IsD["edge_length"])

    # Short path between two known adjacent vertices
    start_vertex, end_vertex = 1396, 1393
    path = find_vertex_path(graph, start_vertex, end_vertex)

    assert isinstance(path, np.ndarray)
    assert path[0] == start_vertex
    assert path[-1] == end_vertex
    assert int(path.sum()) == 2789

    # Longer path across a known section
    target_se = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    target_sw = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    model_se = target_se.to_model_station(ds_IsD)
    model_sw = target_sw.to_model_station(ds_IsD)
    path_long = find_vertex_path(graph, model_se.vertex, model_sw.vertex)

    assert path_long[0] == model_se.vertex
    assert path_long[-1] == model_sw.vertex
    assert int(path_long.sum()) == 55795


def test_vertex_path_to_edge_path(ispy_grid):
    ds_IsD = ispy_grid

    target_se = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    target_sw = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    model_se = target_se.to_model_station(ds_IsD)
    model_sw = target_sw.to_model_station(ds_IsD)
    section = Section("Test", model_se, model_sw, ds_IsD, section_type="shortest")

    edge_path = vertex_path_to_edge_path(ds_IsD, section.vertex_path)

    assert isinstance(edge_path, xr.DataArray)
    assert edge_path.size == section.vertex_path.size - 1
    assert int(np.sum(edge_path.values)) == 143159


def test_orientation_along_path(ispy_grid):
    ds_IsD = ispy_grid

    target_sw = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    target_mid = TargetStation("Mid Corner", -81.0, -23.219, boundary=False)
    target_se = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    model_sw = target_sw.to_model_station(ds_IsD)
    model_mid = target_mid.to_model_station(ds_IsD)
    model_se = target_se.to_model_station(ds_IsD)

    sec_sw_mid = Section("SW to Mid", model_sw, model_mid, ds_IsD, section_type="shortest")
    sec_mid_se = Section("Mid to SE", model_mid, model_se, ds_IsD, section_type="shortest")
    sec_se_sw = Section("SE to SW", model_se, model_sw, ds_IsD, section_type="shortest")
    region = Region("Test Region", [sec_sw_mid, sec_mid_se, sec_se_sw], ds_IsD)

    orientation = orientation_along_path(ds_IsD, region.vertex_circuit, region.edge_circuit)

    assert isinstance(orientation, xr.DataArray)
    assert orientation.size == 20
    assert set(orientation.values.tolist()) == {-1.0, 1.0}
    assert int(np.sum(orientation.values)) == 0
