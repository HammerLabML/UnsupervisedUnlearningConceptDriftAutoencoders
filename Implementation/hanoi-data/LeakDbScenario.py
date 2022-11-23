import os
import pandas as pd
import numpy as np


class Scenario():
    """Class for accessing a scenario from the LeakDb dataset.

    The :class:`Scenario` class loads/parses a given scenario from the LeakDb dataset.

    Parameters
    ----------
    scenario_id : `str`
        ID/Number of the scenario.
    path_to_leakdb : `str`
        Path to the LeakDb folder containing all scenarios -- the network (e.g. either Hanoi or Net1) must be selected in this path!

    Attributes
    ----------
    labels : `numpy.ndarray`
        One dimensional array containing the label ('1' if at least one leakage is present and '0' otherwise) for each point in time -- first axis is time.
    demands : `pandas.DataFrame`
        Data frame containing the demands at all nodes for all time points. Each row corresponds to one time point and nodes are arranged as columns. See attribute `node_ids` for the column names.
    flows : `pandas.DataFrame`
        Data frame containing the flows at all links for all time points. Each row corresponds to one time point and links are arranged as columns. See attribute `link_ids` for the column names.
    pressure : `pandas.DataFrame`
        Data frame containing the pressures at all nodes for all time points. Each row corresponds to one time point and nodes are arranged as columns. See attribute `node_ids` for the column names.
    node_ids : `list`
        List of node IDs.
    link_ids : `list`
        List of link IDs.
    leaky_nodes_ids : `list`
        List of nodes that have a leak at some point in time.
    leaky_times : `list`
        List of all time points at which at least one leakage is present in this scenario.
    """
    def __init__(self, scenario_id, path_to_leakdb):
        self.scenario_id = scenario_id
        self.path_to_leakdb = path_to_leakdb
        self.path_to_scenario = os.path.join(self.path_to_leakdb, f"Scenario-{scenario_id}/")
        
        self.labels = None
        self.demands = None
        self.flows = None
        self.pressures = None
        self.node_ids = None
        self.link_ids = None
        self.leaky_nodes_ids = None
        self.leaky_times = None
        
        self.__load_scenario()
        
    def __list_all_csv_files(self, directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".csv")]

    def __load_scenario_demands(self):
        path_to_scenario_demands = os.path.join(self.path_to_scenario, "Demands")
        demands_files = self.__list_all_csv_files(path_to_scenario_demands)

        demands_per_node = {}
        for f_in in demands_files:
            node_demands = pd.read_csv(os.path.join(path_to_scenario_demands, f_in), index_col="Index")["Value"].to_numpy().flatten()
            demands_per_node[f_in.replace(".csv", "")] = node_demands

        self.demands = pd.DataFrame(demands_per_node)

    def __load_scenario_flows(self):
        path_to_scenario_flows = os.path.join(self.path_to_scenario, "Flows")
        flows_files = self.__list_all_csv_files(path_to_scenario_flows)

        flows_per_link = {}
        for f_in in flows_files:
            link_flows = pd.read_csv(os.path.join(path_to_scenario_flows, f_in), index_col="Index")["Value"].to_numpy().flatten()
            flows_per_link[f_in.replace(".csv", "")] = link_flows

        self.flows = pd.DataFrame(flows_per_link)

    def __load_scenario_pressures(self):
        path_to_scenario_pressures = os.path.join(self.path_to_scenario, "Pressures")
        pressures_files = self.__list_all_csv_files(path_to_scenario_pressures)

        pressures_per_node = {}
        for f_in in pressures_files:
            node_pressures = pd.read_csv(os.path.join(path_to_scenario_pressures, f_in), index_col="Index")["Value"].to_numpy().flatten()
            pressures_per_node[f_in.replace(".csv", "")] = node_pressures

        self.pressures = pd.DataFrame(pressures_per_node)

    def __load_scenario_leaky_nodes(self):
        path_to_leaks = os.path.join(self.path_to_scenario, "Leaks/")
        self.leaky_nodes_ids = [f.replace("_demand.csv", "") for f in os.listdir(path_to_leaks) if os.path.isfile(os.path.join(path_to_leaks, f)) and f.endswith("demand.csv")]

    def __load_scenario(self):
        self.labels = pd.read_csv(os.path.join(self.path_to_scenario, "Labels.csv"), index_col="Index").to_numpy().flatten()
        self.leaky_times = list(np.where(self.labels > 0.)[0])

        self.__load_scenario_pressures()
        self.__load_scenario_demands()
        self.__load_scenario_flows()
        self.__load_scenario_leaky_nodes()

        self.node_ids = list(self.pressures.columns);self.node_ids.sort(key=lambda z: int(z.replace("Node_", "")))
        self.link_ids = list(self.flows.columns);self.link_ids.sort(key=lambda z: int(z.replace("Link_", "")))
        
    def contains_leaks(self):
        """Checks if this scenario contains any leakages or not.

        Does this scenario contain any leakages?

        Returns
        -------
        `boolean`
            'True' if this scenario contains any leakages and 'False' otherwise.
        """
        return len(self.leaky_nodes_ids) > 0


def get_scenarios_with_without_leakages(path_to_leakdb):
    """Creates a list of all scenarios without leakages and another one with all scenarios containing leakages.

    Parameters
    ----------
    path_to_leakdb : `str`
        Path to the LeakDb folder containing all scenarios -- the network (e.g. either Hanoi or Net1) must be selected in this path!
        
    Returns
    -------
    `(list,list)`
        Tuple of two lists -- first entry containts a list with all scenario IDs of scenarios without any leakages and the second entry contains a list with all scenario IDs of scenarios that contain some leakages.
    """
    df_labels = pd.read_csv(os.path.join(path_to_leakdb, "Labels.csv"))
    scenarios_without_leaks = df_labels[df_labels["Label"] == 0.0].reset_index(drop=True)["Scenario"].to_numpy()
    scenarios_with_leaks = df_labels[df_labels["Label"] == 1.0].reset_index(drop=True)["Scenario"].to_numpy()

    return scenarios_without_leaks, scenarios_with_leaks
