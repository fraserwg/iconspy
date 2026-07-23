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
    # Attributes to test
    ## name, station_a, station_b
    ## section_type
    ## vertex_path, edge_path, edge_orientation, vlon, vlat
    ## _uuidOfHGrid
    
    # Methods to test
    ## to_ispy_section
    ## reverse_section
    
    # Section types to test
    ## shortest, isolat, isolon, great circle, rhumb line, contour
    ## isolat and isolon warnings on random sections
    ## Trigger of not implemented section type
    
    # Should we test the hidden weight functions too?
    
    # Test what happens if we use the same station for both ends
    
    
    ds_IsD = ispy_grid
    target_sw_corner = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    target_se_corner = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    target_b_isolat = TargetStation("Isolat Corner", -81.0, -23.219, boundary=False)
    target_b_isolon = TargetStation("Isolon Corner", -92.592, -10, boundary=False)
    
    model_sw_corner = target_sw_corner.to_model_station(ds_IsD)
    model_se_corner = target_se_corner.to_model_station(ds_IsD)
    model_b_isolat = target_b_isolat.to_model_station(ds_IsD)
    model_b_isolon = target_b_isolon.to_model_station(ds_IsD)
    
    # Great circle
    southern_edge_great_circle = Section(
        "Southern Edge (great circle)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="great circle",
    )
    
    # Shortest
    southern_edge_shortest = Section(
        "Southern Edge (shortest)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="shortest",
    )
    assert str(southern_edge_shortest.name) == "Southern Edge (shortest)"
    assert southern_edge_shortest.station_a == model_se_corner
    assert southern_edge_shortest.station_b == model_sw_corner
    assert southern_edge_shortest.section_type == "shortest"
    assert np.sum(southern_edge_shortest.vertex_path) == 55795
    assert np.sum(southern_edge_shortest.edge_path) == 143159
    assert np.sum(southern_edge_shortest.edge_orientation) == -1
    assert np.isclose(np.sum(southern_edge_shortest.vlon), -992.25523174)
    assert np.isclose(np.sum(southern_edge_shortest.vlat), -240.1950428)
    assert southern_edge_shortest._uuidOfHGrid == ds_IsD.attrs["uuidOfHGrid"]
    
    # Rhumb line
    southern_edge_rhumb_line = Section(
        "Southern Edge (rhumb line)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="rhumb line",
    )
    assert southern_edge_rhumb_line.section_type == "rhumb line"
    assert np.sum(southern_edge_rhumb_line.vertex_path) == 65261
    assert np.sum(southern_edge_rhumb_line.edge_path) == 169617
    assert np.sum(southern_edge_rhumb_line.edge_orientation) == 1
    assert np.isclose(np.sum(southern_edge_rhumb_line.vlon), -1138.66188397)
    assert np.isclose(np.sum(southern_edge_rhumb_line.vlat), -305.57724096)

    # Contour
    southern_edge_contour = Section(
        "Southern Edge (contour)",
        model_se_corner,
        model_sw_corner,
        ds_IsD,
        section_type="contour",
        contour_target=20.0,
        contour_data=ds_IsD["elat"]
    )
    assert southern_edge_contour.section_type == "contour"
    assert np.sum(southern_edge_contour.vertex_path) == 55795
    assert np.sum(southern_edge_contour.edge_path) == 143159
    assert np.sum(southern_edge_contour.edge_orientation) == -1
    assert np.isclose(np.sum(southern_edge_contour.vlon), -992.25523174)
    assert np.isclose(np.sum(southern_edge_contour.vlat), -240.1950428)
    
    ## Contour warnings and errors
    with pytest.raises(ValueError):
        _ = Section(
            "contour no data or target",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="contour",
        )
    with pytest.raises(ValueError):
        _ = Section(
            "contour no target",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="contour",
            contour_data=ds_IsD["elat"],
        )
    with pytest.raises(ValueError):
        _ = Section(
            "contour no data",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="contour",
            contour_target=20.0,
        )
    with pytest.raises(ValueError):
        _ = Section(
            "contour wrong shape data",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="contour",
            contour_target=20.0,
            contour_data=ds_IsD["vlat"]
        )
    with pytest.warns(UserWarning):
        _ = Section(
            "not contour but data provided",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="rhumb line",
            contour_target=1000.0,
            contour_data=ds_IsD["elat"]
        )

    # Isolat
    section_isolat = Section(
        "Isolat",
        model_sw_corner,
        model_b_isolat,
        ds_IsD,
        section_type="isolat",
    )
    assert section_isolat.section_type == "isolat"
    assert np.sum(section_isolat.vertex_path) == 36812
    assert np.sum(section_isolat.edge_path) == 90122
    assert np.sum(section_isolat.edge_orientation) == 3
    assert np.isclose(np.sum(section_isolat.vlon), -696.0658456720239)
    assert np.isclose(np.sum(section_isolat.vlat), -185.2302160361028)
    
    ## Isolat warning
    with pytest.warns(UserWarning):
        southern_edge_isolat_warn = Section(
            "Southern Edge (shortest)",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="isolat",
        )

    # Isolon
    section_isolon = Section(
        "Isolon",
        model_sw_corner,
        model_b_isolon,
        ds_IsD,
        section_type="isolon",
    )
    assert section_isolon.section_type == "isolon"
    assert np.sum(section_isolon.vertex_path) == 37010
    assert np.sum(section_isolon.edge_path) == 90603
    assert np.sum(section_isolon.edge_orientation) == -3
    assert np.isclose(np.sum(section_isolon.vlon), -740.8854456901096)
    assert np.isclose(np.sum(section_isolon.vlat), -138.03104997609708)
    
    ## Isolon warning
    with pytest.warns(UserWarning):
        southern_edge_isolon_warn = Section(
            "Southern Edge (shortest)",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="isolon",
        )
        
    with pytest.raises(NotImplementedError):
        _ = Section(
            "not implemented section type",
            model_se_corner,
            model_sw_corner,
            ds_IsD,
            section_type="something weird",
        )

    ## Check methods
    ### Check reverse_section method
    southern_edge_shortest_twice_reversed = southern_edge_shortest.reverse_section().reverse_section()
    assert southern_edge_shortest_twice_reversed.station_a == southern_edge_shortest.station_a
    assert southern_edge_shortest_twice_reversed.station_b == southern_edge_shortest.station_b
    assert np.all(southern_edge_shortest_twice_reversed.vertex_path == southern_edge_shortest.vertex_path)
    assert np.all(southern_edge_shortest_twice_reversed.edge_path == southern_edge_shortest.edge_path)
    assert np.all(southern_edge_shortest_twice_reversed.edge_orientation == southern_edge_shortest.edge_orientation)
    assert np.all(southern_edge_shortest_twice_reversed.vlon == southern_edge_shortest.vlon)
    assert np.all(southern_edge_shortest_twice_reversed.vlat == southern_edge_shortest.vlat)
    assert southern_edge_shortest_twice_reversed._uuidOfHGrid == southern_edge_shortest._uuidOfHGrid
    
    ### Check to_ispy_section method
    southern_edge_shortest_ispy_section = southern_edge_shortest.to_ispy_section()
    
def test_region(ispy_grid):
    pass
