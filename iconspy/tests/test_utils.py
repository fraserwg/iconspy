import pytest
import xarray as xr
from iconspy.utils import convert_tgrid_data, _assert_IsD_compatible
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
