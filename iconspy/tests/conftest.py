import pytest
from pathlib import Path
import xarray as xr
import iconspy as ispy

@pytest.fixture()
def raw_grid():
    cur_dir = Path(__file__).parent.resolve()
    grid_path = cur_dir / "test_data/icon_grid_0014_R02B04_O.nc"

    if not grid_path.exists():
        import requests
        grid_download_link = "http://icon-downloads.mpimet.mpg.de/grids/public\
            /mpim/0014/icon_grid_0014_R02B04_O.nc"
        try:
            r = requests.get(grid_download_link, allow_redirects=True)
            with open(grid_path, "wb") as grid_file:
                grid_file.write(r.content)
        except:
            raise FileNotFoundError(f"{grid_path} does not exist and unable to \
                download it")

    ds_grid = xr.open_dataset(grid_path)
    return ds_grid

@pytest.fixture()
def ispy_grid(raw_grid):
    return ispy.convert_tgrid_data(raw_grid, pyic_kwargs={"old_dim_behaviour": False})


@pytest.fixture()
def boundary_target_station():
    return ispy.TargetStation("Fram Strait West", -14, 80)


@pytest.fixture()
def wet_target_station():
    return ispy.TargetStation("Fram Strait Central", 1, 80, boundary=False)


@pytest.fixture()
def wet_model_station(wet_target_station, ispy_grid):
    return wet_target_station.to_model_station(ispy_grid)

@pytest.fixture()
def boundary_model_station(boundary_target_station, ispy_grid):
    return boundary_target_station.to_model_station(ispy_grid)