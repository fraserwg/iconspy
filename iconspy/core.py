import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from .utils import (
    create_connectivity_matrix,
    create_boundary_connectivity_matrix,
    setup_figure_area,
    find_vertex_path,
    vertex_path_to_edge_path,
    orientation_along_path,
    _assert_IsD_compatible,
)

from .kdtree import (
    find_wet_vertex,
    find_boundary_vertex,
)

import datetime
import copy as cp
import networkx as nx
import shapely
from collections import OrderedDict

class _Name:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_Name({self.name})"

    def __str__(self):
        return f"{self.name}"
    
    def strip_white_space(self):
        return self.name.replace(" ", "")

class TargetStation:
    """Represents a target station

    Parameters
    ----------
    name : str
        Name of the target station
    lon : float
        Longitude of the target station
    lat : float
        Latitude of the target station
    boundary : bool, optional
        Whether the station should be sited on a boundary point or not, by default True
        
    Attributes
    ----------
    name : str
        Name of the target station
    target_lon : float
        Longitude of the target station
    target_lat : float
        Latitude of the target station
    boundary : bool
        Whether the station should be sited on a boundary point or not
        
    Example
    -------
    >>> import iconspy as ispy
    >>> target_station = ispy.TargetStation("My Station", 10.0, 20.0)
    >>> target_station.plot()
    """
    def __init__(self, name, lon, lat, boundary=True):
        self.name = _Name(name)
        self.target_lon = lon
        self.target_lat = lat
        self.boundary = boundary

    def to_model_station(self, ds_IsD):
        """Converts the target station to a model station

        Parameters
        ----------
        ds_IsD : xarray.Dataset
            A dataset contining the model grid information, having been
            operated on by iconspy.convert_tgrid_data()

        Returns
        -------
        iconspy.ModelStation
            A model station object representing the closest model grid point
            to the target station
        """
        if self.boundary:
            station = self.to_boundary_model_station(ds_IsD)
        else:
            station = self.to_wet_model_station(ds_IsD)
        return station
    
    def to_boundary_model_station(self, ds_IsD):
        """Converts the target station to a dry model station regardless of boundary

        Parameters
        ----------
        ds_IsD : xarray.Dataset
            A dataset contining the model grid information, having been
            operated on by iconspy.convert_tgrid_data()

        Returns
        -------
        iconspy.BoundaryModelStation
            A model station object representing the closest dry model grid point
            to the target station
        """
        return BoundaryModelStation(self, ds_IsD)


    def to_wet_model_station(self, ds_IsD):
        """Converts the target station to a wet model station regardless of boundary

        Parameters
        ----------
        ds_IsD : xarray.Dataset
            A dataset contining the model grid information, having been
            operated on by iconspy.convert_tgrid_data()

        Returns
        -------
        iconspy.WetModelStation
            A model station object representing the closest wet model grid point
            to the target station
        """
        return WetModelStation(self, ds_IsD)

    def plot(self, ax=None, proj=None, extent=None,
             coastlines=True, gridlines=True,
            ):
        """plot the target station on a map

        Parameters
        ----------
        ax : matplotlib.axes, optional
            axis to plot on, by default None
        proj : ccrs.Projection, optional
            projection of map to be plotted, by default None
        extent : arraylike, optional
            extent of plot, by default None
        coastlines : bool, optional
            whether to overlay coastlines, by default True
        gridlines : bool, optional
            whether to overlay lines of latitude and longitude, by default True
        """
        fig, ax = setup_figure_area(ax=ax, proj=proj, extent=extent, coastlines=coastlines, gridlines=gridlines)

        ax.plot(
            self.target_lon,
            self.target_lat,
            transform=ccrs.PlateCarree(),
            marker="x",
            label=self.name
        )


class ModelStation:
    """Represents a station on the model grid
    
    Normally created using iconspy.TargetStation.to_model_station()
    
    Parameters
    ----------
    target_station : iconspy.TargetStation
        The target station object that this model station is based on
    
    Attributes
    ---------- 
    name : str
        Name of the target station
    vertex : int
        The vertex number of the model grid point
    model_lon : float
        Longitude of the model grid point
    model_lat : float
        Latitude of the model grid point
    """
    def __init__(self, target_station):
        self.target_station = target_station
        self.name = target_station.name
        self.vertex = None
        self.model_lon = None
        self.model_lat = None 

    def __repr__(self):
        return f"ModelStation({self.name})"

    def plot(self, ax=None, proj=None, extent=None,
             coastlines=True, gridlines=True,
            ):

        fig, ax = setup_figure_area(ax=ax, proj=proj, extent=extent, coastlines=coastlines, gridlines=gridlines)

        ax.scatter(
            self.model_lon,
            self.model_lat,
            transform=ccrs.PlateCarree(),
            marker="o",
            label=self.name,
        )


class WetModelStation(ModelStation):
    def __init__(self, target_station, ds_IsD):
        _assert_IsD_compatible(ds_IsD)
        
        if target_station.boundary == True:
            raise ValueError("target station indicates the model station should be on a boundary")
        super().__init__(target_station)

        self.vertex = int(find_wet_vertex(
            ds_IsD,
            lon=self.target_station.target_lon,
            lat=self.target_station.target_lat,
        ).values[0])
        
        self.model_lon = float(ds_IsD["vlon"].sel(vertex=self.vertex).values)
        self.model_lat = float(ds_IsD["vlat"].sel(vertex=self.vertex).values)


class BoundaryModelStation(ModelStation):
    def __init__(self, target_station, ds_IsD):
        _assert_IsD_compatible(ds_IsD)
        
        if target_station.boundary == False:
            raise ValueError("target station indicates the model station should be wet")
        super().__init__(target_station)

        self.vertex = int(find_boundary_vertex(
            ds_IsD,
            lon=self.target_station.target_lon,
            lat=self.target_station.target_lat,
        ).values[0])
    
        self.model_lon = float(ds_IsD["vlon"].sel(vertex=self.vertex).values)
        self.model_lat = float(ds_IsD["vlat"].sel(vertex=self.vertex).values)


class Section:
    def __init__(self, name, model_station_a, model_station_b, ds_IsD,
                 section_type=None, contour_target=None, contour_data=None):

        _assert_IsD_compatible(ds_IsD)

        self.name = _Name(name)
        self.station_a = model_station_a
        self.station_b = model_station_b
        self.section_type = section_type
        self.vertex_path = None
        self.edge_path = None
        self.edge_orientation = None

        vertex_graph = self.compute_vertex_graph(ds_IsD, contour_target, contour_data)

        vertex_path_np = find_vertex_path(vertex_graph, self.station_a.vertex, self.station_b.vertex)
        self.vertex_path = ds_IsD["vertex"].sel(vertex=vertex_path_np).rename({"vertex": "step_in_path_v"})
        
        self.vlon = ds_IsD["vlon"].sel(vertex=self.vertex_path)
        self.vlat = ds_IsD["vlat"].sel(vertex=self.vertex_path)

        self.edge_path = vertex_path_to_edge_path(ds_IsD, self.vertex_path)

    def __repr__(self):
        return f"Section({self.name}, {self.station_a.name}, {self.station_b.name}, {self.section_type})"

    def to_ispy_section(self, fpath, dryrun=False):
        print(f"Output will be saved to {fpath}")
        ds_path = xr.Dataset()
        ds_path["edge_path"] = self.edge_path
        ds_path["vertex_path"] = self.vertex_path
        ds_path["path_orientation"] = self.edge_orientation
        # ds_path.attrs["grid_path"] = str(icon_control["2d_grid"].metadata)
        # ds_path.attrs["author"] = "Fraser Goldsworth: frasergocean[at]gmail.com"
        ds_path.attrs["date"] = str(datetime.datetime.now())[:19]
        # ds_path.attrs["script"] = str(sba.base_path / "src/section-construction/v2-section-construction.ipynb")
        # ds_path.attrs["version"] = f"sba-proj: {sba.get_git_commit_hash()}"

        if dryrun:
            print("Not saving as dryrun=True")
        else:
            ds_path.to_netcdf(fpath)

        return ds_path

    def set_pyic_orientation_along_path(self, ds_IsD):
        """Calculate the orientation of the edges along the path
        
        Add the edge orientation object to the section object using code from
        pyicon.
        
        Parameters
        ----------
        self : iconspy.Section
            The section you wish to calculate the edge orientation for

        ds_IsD : xarray.Dataset
            A dataset containing the model grid information, having been
            operated on by iconspy.convert_tgrid_data()
        
        Notes
        -----
        Original code from pyicon under the MIT license.
        Modified by Fraser Goldsworth on 18.06.2025.
        See LICENSE file for more information.

        """
        
        _assert_IsD_compatible(ds_IsD)
        
        ie_list = self.edge_path
        iv_list = self.vertex_path
        
        or_list = np.zeros((ie_list.size))
        for nn in range(ie_list.size):
            iel = ds_IsD.edges_of_vertex[iv_list[nn],:]==ie_list[nn]
            or_list[nn] = ds_IsD.edge_orientation[iv_list[nn], iel]
            
        orientation = xr.ones_like(self.edge_path) * or_list
        
        self.edge_orientation = orientation

    def plot(self, ax=None, proj=None, extent=None,
             coastlines=True, gridlines=True,
            ):

        fig, ax = setup_figure_area(ax=ax, proj=proj, extent=extent, coastlines=coastlines, gridlines=gridlines)

        ax.plot(
            self.vlon,
            self.vlat,
            transform=ccrs.PlateCarree(),
            label=self.name,
        )
    
    def compute_vertex_graph(self, ds_IsD, contour_target=None, contour_data=None):
        _assert_IsD_compatible(ds_IsD)
        
        if self.section_type == "shortest":
            weights = ds_IsD["edge_length"]
        elif self.section_type == "isolat":
            raise NotImplementedError("section type requested is not implemented")
        elif self.section_type == "isolon":
            raise NotImplementedError("section type requested is not implemented")
        elif self.section_type == "great circle":
            weights = self.great_circle_weights(ds_IsD)
        elif self.section_type == "lat lon straight line":
            weights = self.lat_lon_as_cartesian_weights(ds_IsD)
        elif self.section_type == "contour":
            weights = self.contour_weights(ds_IsD, contour_target, contour_data)
        else:
            raise NotImplementedError("section type requested is not implemented")

        vertex_graph = create_connectivity_matrix(ds_IsD, weights)

        return vertex_graph
    

    def contour_weights(self, ds_IsD, contour_target, contour_data):
        weights = abs(contour_data - contour_target)
        return weights
    
    def reverse_section(self):
        reversed_self = cp.deepcopy(self)
        reversed_self.station_a, reversed_self.station_b = self.station_b, self.station_a
        reversed_self.vertex_path = self.vertex_path[::-1]
        if self.edge_path is not None:
            reversed_self.edge_path = self.edge_path[::-1]
        if self.edge_orientation is not None:
            reversed_self.edge_orientation = self.edge_orientation[::-1]
        return reversed_self
    
    def great_circle_weights(self, ds_IsD, angular_distance=True):
        _assert_IsD_compatible(ds_IsD)
        
        station_a_cart = ds_IsD["vert_cart_vec"].sel(vertex=self.station_a.vertex)
        station_b_cart = ds_IsD["vert_cart_vec"].sel(vertex=self.station_b.vertex)
        
        great_circle_normal = np.cross(station_a_cart, station_b_cart)
        great_circle_normal_for_edges = np.cross(ds_IsD["edge_cart_vec"], great_circle_normal)
        closest_great_circle_point = np.cross(great_circle_normal, great_circle_normal_for_edges) * xr.ones_like(ds_IsD["edge_cart_vec"])
        
        edge_great_circle_vec = closest_great_circle_point - ds_IsD["edge_cart_vec"]

        if angular_distance:
            adotb = xr.dot(closest_great_circle_point, ds_IsD["edge_cart_vec"], dim="cart")
            moda = xr.apply_ufunc(
                np.linalg.norm,
                closest_great_circle_point,
                input_core_dims=[["cart"]],
                kwargs={"axis": 1}
            )
                
            modb = xr.apply_ufunc(
                np.linalg.norm,
                ds_IsD["edge_cart_vec"],
                input_core_dims=[["cart"]],
                kwargs={"axis": 1}
            )
    
            costheta = adotb / moda / modb
            costheta = costheta.where(costheta < 1, other=1)
            costheta = costheta.where(costheta > -1, other=-1)
            weights = np.arccos(costheta)
        else:                
            weights = xr.apply_ufunc(
                np.linalg.norm,
                edge_great_circle_vec,
                input_core_dims=[["cart"]],
                kwargs={"axis": 1}
            )

        return weights


    def lat_lon_as_cartesian_weights(self, ds_IsD):
        x1 = ds_IsD["vlon"].sel(vertex=self.station_a.vertex).squeeze()
        x2 = ds_IsD["vlon"].sel(vertex=self.station_b.vertex).squeeze()
        
        y1 = ds_IsD["vlat"].sel(vertex=self.station_a.vertex).squeeze()
        y2 = ds_IsD["vlat"].sel(vertex=self.station_b.vertex).squeeze()
        
        x0s = ds_IsD["elon"]
        y0s = ds_IsD["elat"]
        
        numerator = np.abs((y2 - y1) * x0s - (x2 - x1) * y0s + x2 * y1 - y2 * x1)
        denominator = np.sqrt(np.square(y2 - y1) + np.square(x2 - x1))
        weights = numerator / denominator

        return weights

class LandSection(Section):
    def __repr__(self):
        return f"LandSection({self.name}, {self.station_a.name}, {self.station_b.name}, {self.section_type})"
    
    def compute_vertex_graph(self, ds_IsD, contour_target=None, contour_data=None):
        if self.section_type != "shortest":
            raise ValueError(f"LandSection should have section type of 'shortest', not {self.section_type}")

        vertex_graph = create_boundary_connectivity_matrix(ds_IsD, weight_type="distance")

        return vertex_graph

class _ReconstructedSection(Section):
    def __init__(self, section, vertex_path, edge_path, edge_orientation, ds_IsD):
        self.name = section.name
        self.station_a = section.station_a
        self.station_b = section.station_b
        self.section_type = section.section_type
        self.vertex_path = vertex_path
        self.edge_path = edge_path
        self.edge_orientation = edge_orientation
        self.vlon = ds_IsD["vlon"].sel(vertex=self.vertex_path)
        self.vlat = ds_IsD["vlat"].sel(vertex=self.vertex_path)

    def __repr__(self):
        return f"ReconstructedSection({self.name}, {self.station_a.name}, {self.station_b.name}, {self.section_type})"


class CombinedSection(Section):
    def __init__(self, name, section_list, ds_IsD):
        # Want to be able to combine two section into one.

        # Check the sections connect...
        assert section_list[0].vertex_path.isel(step_in_path_v=-1) == section_list[1].vertex_path.isel(step_in_path_v=0)
        self.name = name
        self.station_a = section_list[0].station_a
        self.station_b = section_list[-1].station_b
        
        if section_list[0].section_type == section_list[1].section_type:
            section_type = section_list[0].section_type
        else:
            section_type = "mixed"
        self.section_type = section_type
        
        self.vertex_path = xr.concat([section_list[0].vertex_path.isel(step_in_path_v=slice(0, -1)), section_list[1].vertex_path], dim="step_in_path_v")
        self.edge_path = xr.concat([section_list[0].edge_path, section_list[1].edge_path], dim="step_in_path")
        self.edge_orientation = None  # We can't be sure that the two sections have the same sign convention
        self.vlon = ds_IsD["vlon"].sel(vertex=self.vertex_path)
        self.vlat = ds_IsD["vlat"].sel(vertex=self.vertex_path)   



class Region:
    def __init__(self, name, section_list, ds_IsD, test=False, manual_order=False):
        self.name = _Name(name)
        self.section_list = None
        
        _assert_IsD_compatible(ds_IsD)
        
        # Order the sections provided
        if not manual_order:
            section_order = self.calculate_section_order(section_list)
            self.order_sections(section_order, section_list)
            # Check the sections have been successfuly ordered
            assert self.calculate_section_order(self.section_list) == list(range(len(self.section_list)))
        else:
            self.section_list = section_list

        # Get the vertex, edge and orientation xr.DataArrays
        self.vertex_circuit = self.calculate_vertex_circuit(ds_IsD)
        if not test:
            self.edge_circuit = vertex_path_to_edge_path(ds_IsD, self.vertex_circuit)
            self.path_orientation = orientation_along_path(ds_IsD, self.vertex_circuit, self.edge_circuit)
            self.contained_cells = self.calculate_contained_cells(ds_IsD)

    def __repr__(self):
        return f"Region({self.name}, {self.section_list})"

    def to_pyicon_section(self, fpath):
        raise NotImplementedError("Method not yet implemented")


    def to_ispy_section(self, fpath, dryrun=False):
        print(f"Output will be saved to {fpath}")
            
        ds_path = xr.Dataset()
        ds_path["edge_path"] = self.edge_circuit
        ds_path["vertex_path"] = self.vertex_circuit
        ds_path["path_orientation"] = self.path_orientation
        ds_path["contained_cells"] = self.contained_cells
        # ds_path.attrs["grid_path"] = str(icon_control["2d_grid"].metadata)
        # ds_path.attrs["author"] = "Fraser Goldsworth: frasergocean[at]gmail.com"
        ds_path.attrs["date"] = str(datetime.datetime.now())[:19]
        # ds_path.attrs["script"] = str(sba.base_path / "src/section-construction/v2-section-construction.ipynb")
        # ds_path.attrs["version"] = f"sba-proj: {sba.get_git_commit_hash()}"

        if dryrun:
            print("Not saving as dryrun=True")
        else:
            ds_path.to_netcdf(fpath)

        return ds_path

        
    def calculate_contained_cells(self, ds_IsD):
        ring_coords = xr.concat(
            (self.vertex_circuit["vlon"], self.vertex_circuit["vlat"]), dim="cart"
        ).transpose(..., "cart")
        
        enclosed_area = shapely.polygons(ring_coords)
        cell_points = shapely.points(ds_IsD["clon"], ds_IsD["clat"])
        cell_idxs, = np.where(enclosed_area.contains(cell_points))
        contained_cells = ds_IsD["cell"].isel(cell=cell_idxs)
        return contained_cells
    

    def plot(self, ax=None, proj=None, extent=None,
             coastlines=True, gridlines=True,
            ):

        fig, ax = setup_figure_area(ax=ax, proj=proj, extent=extent, coastlines=coastlines, gridlines=gridlines)

        ax.plot(
            self.vertex_circuit["vlon"],
            self.vertex_circuit["vlat"],
            transform=ccrs.PlateCarree()
        )

        ax.fill(
            self.vertex_circuit["vlon"],
            self.vertex_circuit["vlat"],
            transform=ccrs.PlateCarree(),
            alpha=0.5,
            label=self.name
        )
    
    def calculate_vertex_circuit(self, ds_IsD):
        # Create the vertex circuit by combining the sections
        vertex_circuit = np.hstack([section.vertex_path for section in self.section_list])
        
        # Remove duplicates and dead branches
        edge_list = np.concatenate((vertex_circuit[1:, None], vertex_circuit[:-1, None]), axis=1)
        graph = nx.from_edgelist(edge_list, create_using=nx.Graph)
    
        simple_cycles = list(nx.simple_cycles(graph))
        cycle_length = np.array([len(cycle) for cycle in simple_cycles])
        idx_of_longest_cycle = cycle_length.argmax()
    
        vertex_circuit_no_reps = simple_cycles[idx_of_longest_cycle]
    
        # Do some error checking
        _, reps_of_vertex = np.unique(vertex_circuit_no_reps, return_counts=True)
        assert np.all(reps_of_vertex == 1)
        
        vertex_circuit = np.array(vertex_circuit_no_reps)
        vertex_circuit = np.append(vertex_circuit, vertex_circuit[0])

        # Convert to xarray objects
        vertex_path_xr = ds_IsD["vertex"].sel(vertex=vertex_circuit)
        vertex_path_xr = vertex_path_xr.assign_coords(step_in_path=("vertex", np.arange(vertex_circuit.size))).swap_dims(vertex="step_in_path_v")
        
        return vertex_path_xr

    
    def order_sections(self, section_order, section_list):
        ordered_section_list = []
        for section_index in section_order:
            if section_index >= 0:
                ordered_section_list += [section_list[section_index]]
            else:
                section_index *= -1
                ordered_section_list += [section_list[section_index].reverse_section()]

        self.section_list = ordered_section_list    


    def calculate_section_order(self, section_list):
        """ Given a list of sections, calculates the
        order the sections should be in.

        Notes
        -----
        This function is nigh on unreadable.
        Sorry.
        """
        # Make a list of the start and end vertices
        starts = {i: section.vertex_path[0] for i, section in enumerate(section_list)}
        ends = {i: section.vertex_path[-1] for i, section in enumerate(section_list)}
    
        # We're going to start by saying the zeroth section is in the right place.
        # We will try and join the other sections onto the end of the zeroth
        section_order = [0]
        current_section = 0
        current_end = ends[0]
        starts.pop(0)
        ends.pop(0)
    
        # We have len(ends) more lists to order
        for i in range(len(ends)):
            found = False
            if not found:
                # Check if the end of the current section is
                # in the start lists
                for section_index in starts:
                    if starts[section_index] == current_end:
                        section_order += [1 * section_index]
                        current_section = section_index
                        if i != len(ends):
                            current_end = ends[section_index]
                        found = True
                        break
        
            if not found:
                # Maybe the section is the wrong way around?
                # Check the ends list
                for section_index in ends:
                    if ends[section_index] == current_end:
                        section_order += [-1 * section_index]
                        current_section = -section_index
                        if i != len(ends):
                            current_end = starts[section_index]
                        found = True
                        break
    
            if not found:
                raise ValueError("Cannot order the list provided")
            if i != len(ends):
                # If we found the next list item in the ends or the starts we do things differently
                if current_section < 0:
                    ends.pop(-current_section)
                    starts.pop(-current_section)
                else:
                    starts.pop(current_section)
                    ends.pop(current_section)
                
        return section_order

    def extract_sections_from_region(self, ds_IsD):
        reconstructed_section_dict = OrderedDict()
        for section in self.section_list:
            amended_vertex_path = self.vertex_circuit.isel(
                step_in_path_v=np.isin(self.vertex_circuit, section.vertex_path)
            )
    
            edge_path_mask = np.isin(self.edge_circuit, section.edge_path)
            amended_edge_path = self.edge_circuit.isel(
                step_in_path=edge_path_mask
            )
    
            amended_edge_orientation = self.path_orientation.isel(
                step_in_path=edge_path_mask
            )
    
            reconstructed_section_dict[str(section.name)] = (
                _ReconstructedSection(section, amended_vertex_path, amended_edge_path, amended_edge_orientation, ds_IsD)
            )
        return reconstructed_section_dict
