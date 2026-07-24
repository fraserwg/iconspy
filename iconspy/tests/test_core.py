import pytest
from collections import OrderedDict
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
    # Methods still to test
    ## to_ispy_section
    ## weight functions (hidden)
    # Should we test the hidden weight functions too? Their testing is implicit
    # in the section tests. But maybe we should test the types they return?
    
    # Test what happens if we use the same station for both ends
    
    # Prepare some model stations
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
    ## Only need to test these attributes once
    assert str(southern_edge_shortest.name) == "Southern Edge (shortest)"
    assert southern_edge_shortest.station_a == model_se_corner
    assert southern_edge_shortest.station_b == model_sw_corner
    ## Check these attributes on each section type
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
    assert isinstance(southern_edge_shortest_ispy_section, xr.Dataset)


def test_CombinedSection(ispy_grid):
    ds_IsD = ispy_grid

    target_sw = TargetStation("SW Corner", -92.592, -23.219, boundary=False)
    target_mid = TargetStation("Mid Corner", -81.0, -23.219, boundary=False)
    target_se = TargetStation("SE Corner", -70.285, -18.491, boundary=True)
    model_sw = target_sw.to_model_station(ds_IsD)
    model_mid = target_mid.to_model_station(ds_IsD)
    model_se = target_se.to_model_station(ds_IsD)

    section_1 = Section("SW to Mid", model_sw, model_mid, ds_IsD, section_type="shortest")
    section_2 = Section("Mid to SE", model_mid, model_se, ds_IsD, section_type="shortest")

    # Instantiation and type check
    combined = CombinedSection("Combined", [section_1, section_2], ds_IsD)
    assert isinstance(combined, CombinedSection)
    assert isinstance(combined, Section)

    ## Check attributes
    assert str(combined.name) == "Combined"
    assert combined.station_a == model_sw
    assert combined.station_b == model_se
    assert combined.section_type == "shortest"
    assert np.sum(combined.vertex_path) == 55806
    assert np.sum(combined.edge_path) == 143165
    assert np.sum(combined.edge_orientation) == 1
    assert combined._uuidOfHGrid == ds_IsD.attrs["uuidOfHGrid"]

    ## Check mixed section_type
    section_1b = Section("SW to Mid rhumb", model_sw, model_mid, ds_IsD, section_type="rhumb line")
    section_2b = Section("Mid to SE shortest", model_mid, model_se, ds_IsD, section_type="shortest")
    combined_mixed = CombinedSection("Mixed", [section_1b, section_2b], ds_IsD)
    assert combined_mixed.section_type == "mixed"

    ## Check methods
    assert isinstance(combined.to_ispy_section(), xr.Dataset)

    ## Check ValueError when sections don't connect
    section_3 = Section("SE to SW", model_se, model_sw, ds_IsD, section_type="shortest")
    with pytest.raises(ValueError):
        CombinedSection("Bad", [section_1, section_3], ds_IsD)

    # 3-section combination
    target_ne = TargetStation("NE Corner", -70.285, -10.0, boundary=True)
    model_ne = target_ne.to_model_station(ds_IsD)
    section_3b = Section("SE to NE", model_se, model_ne, ds_IsD, section_type="shortest")

    combined3 = CombinedSection("Three Section Combined", [section_1, section_2, section_3b], ds_IsD)
    assert str(combined3.name) == "Three Section Combined"
    assert combined3.station_a == model_sw
    assert combined3.station_b == model_ne
    assert combined3.section_type == "shortest"
    assert np.sum(combined3.vertex_path) == 65341
    assert np.sum(combined3.edge_path) == 169928
    assert np.sum(combined3.edge_orientation) == 1

    # Wrong order with one section reversed raises ValueError
    ## Correct order is [section_1, section_2, section_3b]; here section_3b is reversed
    ## and placed first, breaking both ordering and direction
    with pytest.raises(ValueError):
        CombinedSection("Bad order", [section_3b.reverse_section(), section_1, section_2], ds_IsD)

    # Wrong order with no sections reversed also raises ValueError
    with pytest.raises(ValueError):
        CombinedSection("Wrong order", [section_2, section_3b, section_1], ds_IsD)


def test_LandSection(ispy_grid):
    ds_IsD = ispy_grid

    target_b1 = TargetStation("Boundary 1", -14, 80, boundary=True)
    target_b2 = TargetStation("Boundary 2", 5, 78, boundary=True)
    target_w1 = TargetStation("Water 1", -92.592, -23.219, boundary=False)
    target_w2 = TargetStation("Water 2", -14, 80, boundary=None)
    
    model_b1 = target_b1.to_model_station(ds_IsD)
    model_b2 = target_b2.to_model_station(ds_IsD)
    model_w1 = target_w1.to_model_station(ds_IsD)
    model_w2 = target_w2.to_model_station(ds_IsD)

    # Instantiation and type check
    ls = LandSection("Land Section", model_b1, model_b2, ds_IsD)
    assert isinstance(ls, LandSection)
    assert isinstance(ls, Section)

    ## Check attributes
    assert str(ls.name) == "Land Section"
    assert ls.station_a == model_b1
    assert ls.station_b == model_b2
    assert ls.section_type == "shortest"
    assert np.sum(ls.vertex_path) == 6983
    assert np.sum(ls.edge_path) == 14737
    assert np.sum(ls.edge_orientation) == 0
    assert ls._uuidOfHGrid == ds_IsD.attrs["uuidOfHGrid"]

    # Antarctica to Greenland
    # Present implementation allows a path to be found but setting wet edges
    # to have 1e6 times the weight of a land edge.
    target_ant = TargetStation("Antarctica", 0, -90, boundary=True)
    target_green = TargetStation("Greenland", -42, 72, boundary=True)
    model_ant = target_ant.to_model_station(ds_IsD)
    model_green = target_green.to_model_station(ds_IsD)

    ls_long = LandSection("Antarctica to Greenland", model_ant, model_green, ds_IsD)
    assert str(ls_long.name) == "Antarctica to Greenland"
    assert ls_long.station_a == model_ant
    assert ls_long.station_b == model_green
    assert np.sum(ls_long.vertex_path) == 598718
    assert np.sum(ls_long.edge_path) == 1690127
    assert np.sum(ls_long.edge_orientation) == 12
    

    # Check starting with wet stations triggers and error
    with pytest.raises(ValueError):
        LandSection("Wet to Wet", model_b1, model_w1, ds_IsD)
    with pytest.raises(ValueError):
        LandSection("Wet to Wet", model_b1, model_w2, ds_IsD)
    with pytest.raises(ValueError):
        LandSection("Wet to Wet", model_w1, model_b2, ds_IsD)
    with pytest.raises(ValueError):
        LandSection("Wet to Wet", model_w2, model_b1, ds_IsD)

def test_region(ispy_grid):
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

    # Instantiation with test=True (skips edge/orientation/cells)
    region_test = Region("Test Region", [sec_sw_mid, sec_mid_se, sec_se_sw], ds_IsD, test=True)
    assert isinstance(region_test, Region)

    ## Check attributes
    assert str(region_test.name) == "Test Region"
    assert region_test._uuidOfHGrid == ds_IsD.attrs["uuidOfHGrid"]
    assert len(region_test.section_list) == 3
    assert region_test.vertex_circuit.size == 21
    assert not hasattr(region_test, "edge_circuit")

    # Full instantiation
    region_full = Region("Full Region", [sec_sw_mid, sec_mid_se, sec_se_sw], ds_IsD)
    assert np.sum(region_full.vertex_circuit) == 97618
    assert np.sum(region_full.edge_circuit) == 260598
    assert np.sum(region_full.path_orientation) == 0
    assert region_full.contained_cells.size == 32

    ## Check methods
    assert isinstance(region_full.to_ispy_section(), xr.Dataset)
    extracted = region_full.extract_sections_from_region(ds_IsD)
    assert isinstance(extracted, OrderedDict)
    assert list(extracted.keys()) == ["SW to Mid", "Mid to SE", "SE to SW"]

    # Disconnected sections should raise ValueError
    target_ne = TargetStation("NE Corner", -70.285, -10.0, boundary=True)
    target_nw = TargetStation("NW Corner", -92.592, -10.0, boundary=True)
    model_ne = target_ne.to_model_station(ds_IsD)
    model_nw = target_nw.to_model_station(ds_IsD)
    sec_ne_nw = Section("NE to NW", model_ne, model_nw, ds_IsD, section_type="shortest")

    with pytest.raises(ValueError):
        Region("Disconnected", [sec_sw_mid, sec_mid_se, sec_ne_nw], ds_IsD, test=True)
