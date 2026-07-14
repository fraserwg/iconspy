import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def generate_sample_region():
    pass

def setup_plot_area_quickstart(ds_IsD):
    Slat, Nlat = -25, 5
    Wlon, Elon = -100, -65

    edges_in_region = ds_IsD["edge"].where(
        (ds_IsD["elon"] > Wlon) * (ds_IsD["elon"] < Elon) * (ds_IsD["elat"] > Slat) * (ds_IsD["elat"] < Nlat)
        , drop=True).astype("int32")


    lons = ds_IsD["vlon"].sel(vertex=ds_IsD["edge_vertices"])
    lats = ds_IsD["vlat"].sel(vertex=ds_IsD["edge_vertices"])

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    for edge in edges_in_region:
        ax.plot(
            lons.isel(edge=edge),
            lats.isel(edge=edge),
            color="black",
            alpha=0.2,
        )

    ax.scatter(
        ds_IsD["vlon"],
        ds_IsD["vlat"],
        s=4,
        transform=ccrs.PlateCarree(),
        color="tab:green"
    )

    ax.set_xlim(Wlon, Elon)
    ax.set_ylim(Slat, Nlat)
    ax.set_aspect("equal")

    ax.coastlines()
    ax.grid(False)
    return fig, ax


def setup_plot_area_mar(ds_IsD):
    Slat, Nlat = -10, 45
    Wlon, Elon = -70, 20

    edges_in_region = ds_IsD["edge"].where(
        (ds_IsD["elon"] > Wlon) * (ds_IsD["elon"] < Elon) * (ds_IsD["elat"] > Slat) * (ds_IsD["elat"] < Nlat)
        , drop=True).astype("int32")


    lons = ds_IsD["vlon"].sel(vertex=ds_IsD["edge_vertices"])
    lats = ds_IsD["vlat"].sel(vertex=ds_IsD["edge_vertices"])

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    for edge in edges_in_region:
        ax.plot(
            lons.isel(edge=edge),
            lats.isel(edge=edge),
            color="black",
            alpha=0.2,
        )

    ax.scatter(
        ds_IsD["vlon"],
        ds_IsD["vlat"],
        s=4,
        transform=ccrs.PlateCarree(),
        color="tab:green"
    )

    ax.set_xlim(Wlon, Elon)
    ax.set_ylim(Slat, Nlat)
    ax.set_aspect("equal")

    ax.coastlines()
    ax.grid(False)
    return fig, ax