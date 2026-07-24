import pytest
from pathlib import Path
import xarray as xr
import iconspy as ispy


def get_ds_tgrid_lr():
    test_data_dir = Path(__file__).parent.resolve() / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    grid_path = test_data_dir / "icon_grid_0036_R02B04_O.nc"


    if not grid_path.exists():
        import requests

        tgrid_swift_url = "https://swift.dkrz.de/v1/dkrz_7fa6baba-db43-4d12-a295-8e3ebb1a01ed/iconspy_test_data/icon_grid_0036_R02B04_O.nc?temp_url_sig=13bad6dab6cbbe7fc81a1d34236e21b3abeda537&temp_url_expires=2036-07-11T08:18:50Z"
        try:
            ds_grid = xr.open_dataset(tgrid_swift_url, engine="h5netcdf")
            ds_grid.to_netcdf(grid_path)
        except:
            raise FileNotFoundError(
                "{grid_path} does not exist and unable to \
                download it"
            )

    ds_grid = xr.open_dataset(grid_path, chunks="auto", engine="h5netcdf")
    return ds_grid


def get_ds_fxgrid_lr():
    test_data_dir = Path(__file__).parent.resolve() / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    fx_grid_path = test_data_dir / "R2B4L40_fx.nc"

    if not fx_grid_path.exists():
        import requests

        fxgrid_swift_url = "https://swift.dkrz.de/v1/dkrz_7fa6baba-db43-4d12-a295-8e3ebb1a01ed/iconspy_test_data/R2B4L40_fx.nc?temp_url_sig=5feaaad2cc6f50595bcf2ba551080fc383fe2980&temp_url_expires=2036-07-11T08:20:05Z"
        try:
            ds_fxgrid = xr.open_dataset(fxgrid_swift_url, engine="h5netcdf")
            ds_fxgrid.to_netcdf(fx_grid_path)
        except:
            raise FileNotFoundError(
                "{fx_grid_path} does not exist and unable to \
                download it from {fxgrid_swift_url}"
            )

    ds_fxgrid = xr.open_dataset(fx_grid_path, chunks="auto", engine="h5netcdf")
    return ds_fxgrid


@pytest.fixture()
def raw_grid():
    return get_ds_tgrid_lr()

@pytest.fixture()
def ispy_grid(raw_grid):
    return ispy.convert_tgrid_data(raw_grid)


@pytest.fixture()
def boundary_target_station():
    return ispy.TargetStation("Fram Strait West", -14, 80)

@pytest.fixture()
def wet_target_station():
    return ispy.TargetStation("Fram Strait Central", 1, 80, boundary=False)

@pytest.fixture()
def any_target_station():
    return ispy.TargetStation("Fram Strait Any", 0, 80, boundary=None)


@pytest.fixture()
def wet_model_station(wet_target_station, ispy_grid):
    return wet_target_station.to_model_station(ispy_grid)

@pytest.fixture()
def boundary_model_station(boundary_target_station, ispy_grid):
    return boundary_target_station.to_model_station(ispy_grid)

@pytest.fixture()
def any_model_station(any_target_station, ispy_grid):
    return any_target_station.to_model_station(ispy_grid)