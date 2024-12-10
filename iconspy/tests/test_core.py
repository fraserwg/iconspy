from iconspy.core import *
from .conftest import raw_grid, ispy_grid, boundary_target_station


def test_TargetStation(ispy_grid):
    fram_strait_west = TargetStation("Fram Strait West", -14, 80)
    isinstance(fram_strait_west.to_model_station(ispy_grid), BoundaryModelStation)
    
    fram_strait_central = TargetStation("Fram Strait Central", 1, 80, boundary=False)
    isinstance(fram_strait_central.to_model_station(ispy_grid), WetModelStation)
    
    
def test_ModelStation(boundary_target_station):
    pass


def test_region(ispy_grid):
    pass
