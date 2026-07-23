import pytest
from iconspy.core import *
from .conftest import raw_grid, ispy_grid, boundary_target_station


def test_TargetStation(ispy_grid):
    # Do a boundary station test
    fram_strait_west = TargetStation("Fram Strait West", -14, 80)
    ## Check type
    assert isinstance(fram_strait_west.to_model_station(ispy_grid), BoundaryModelStation)
    
    ## Check attributes
    assert str(fram_strait_west.name) == "Fram Strait West"
    assert fram_strait_west.target_lon == -14
    assert fram_strait_west.target_lat == 80
    assert fram_strait_west.boundary is True

    ## Check methods
    fram_strait_west.to_model_station(ispy_grid)
    fram_strait_west._to_boundary_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_west._to_any_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_west._to_wet_model_station(ispy_grid)


    # Do a wet station test
    fram_strait_central = TargetStation("Fram Strait Central", 1, 80, boundary=False)
    
    ## Check type
    assert isinstance(fram_strait_central.to_model_station(ispy_grid), WetModelStation)
    
    ## Check attributes
    assert str(fram_strait_central.name) == "Fram Strait Central"
    assert fram_strait_central.target_lon == 1
    assert fram_strait_central.target_lat == 80
    assert fram_strait_central.boundary is False
    
    ## Check methods
    fram_strait_central.to_model_station(ispy_grid)
    fram_strait_central._to_wet_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_central._to_boundary_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_central._to_any_model_station(ispy_grid)
    
    
    # Do a non-guaranteed wet station test
    fram_strait_central = TargetStation("Fram Strait Central", 1, 80, boundary=None)
    
    ## Check type
    assert isinstance(fram_strait_central.to_model_station(ispy_grid), ModelStation)
    
    ## Check attributes
    assert str(fram_strait_central.name) == "Fram Strait Central"
    assert fram_strait_central.target_lon == 1
    assert fram_strait_central.target_lat == 80
    assert fram_strait_central.boundary is None

    ## Check methods
    fram_strait_central.to_model_station(ispy_grid)
    fram_strait_central._to_any_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_central._to_boundary_model_station(ispy_grid)
    with pytest.raises(ValueError):
        fram_strait_central._to_wet_model_station(ispy_grid)

def test_ModelStation(ispy_grid, boundary_target_station, wet_target_station, any_target_station):
    ds_IsD = ispy_grid
    
    # Check any model station
    ## Instantiation and type check
    a = ModelStation(any_target_station, ds_IsD)
    assert isinstance(a, ModelStation)
    
    ## Check attributes
    assert a.target_station == any_target_station
    assert str(a.name) == "Fram Strait Any"
    assert a.vertex == 1393
    assert a.model_lon == 0.9999999999999997
    assert a.model_lat == 80.15757345969898
    assert a._uuidOfHGrid == "5bd948e8-ac1a-11ea-a6b1-d317264fdca9"
    
    
    # Check boundary model station
    ## Instantiation and type check
    b = BoundaryModelStation(boundary_target_station, ds_IsD)
    assert isinstance(b, BoundaryModelStation)
    
    ## Check attributes
    assert b.target_station == boundary_target_station
    assert str(b.name) == "Fram Strait West"
    assert b.vertex == 1396
    assert b.model_lon == -12.141179795983037
    assert b.model_lat == 79.95366451600837
    assert b._uuidOfHGrid == "5bd948e8-ac1a-11ea-a6b1-d317264fdca9"

    # Check wet model station
    ## Instantiation and type check
    c = WetModelStation(wet_target_station, ds_IsD)
    assert isinstance(c, WetModelStation)
    
    ## Check attributes
    assert c.target_station == wet_target_station
    assert str(c.name) == "Fram Strait Central"
    assert c.vertex == 1393
    assert c.model_lon == 0.9999999999999997
    assert c.model_lat == 80.15757345969898
    assert c._uuidOfHGrid == "5bd948e8-ac1a-11ea-a6b1-d317264fdca9"   

    

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
