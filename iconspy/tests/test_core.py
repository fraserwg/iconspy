from iconspy.core import *
from .conftest import raw_grid, ispy_grid, boundary_target_station


def test_TargetStation(ispy_grid):
    fram_strait_west = TargetStation("Fram Strait West", -14, 80)
    assert isinstance(fram_strait_west.to_model_station(ispy_grid), BoundaryModelStation)
    
    fram_strait_central = TargetStation("Fram Strait Central", 1, 80, boundary=False)
    assert isinstance(fram_strait_central.to_model_station(ispy_grid), WetModelStation)
    

def test_ModelStation(ispy_grid, boundary_target_station, wet_target_station):
    ds_IsD = ispy_grid
    
    a = boundary_target_station.to_model_station(ds_IsD)
    assert isinstance(a, BoundaryModelStation)
    
    b = wet_target_station.to_model_station(ds_IsD)
    assert isinstance(b, WetModelStation)
    
    
def test_Section(ispy_grid):
    ds_IsD = ispy_grid
    
    target_sw_corner = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    target_se_corner = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    
    model_sw_corner = target_sw_corner.to_model_station(ds_IsD)
    model_se_corner = target_se_corner.to_model_station(ds_IsD)
    
    # Great circle
    southern_edge_great_circle = Section(
        "Southern Edge (great circle)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="great circle",
    )
    
    # Shortest path
    southern_edge_shortest = Section(
        "Southern Edge (shortest)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="shortest",
    )


def test_region(ispy_grid):
    pass
