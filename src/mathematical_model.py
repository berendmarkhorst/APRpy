import highspy as hp
from .objects import *
import time
import logging
from typing import List, Union, Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def make_model(time_limit: float, logfile: str = "") -> hp.HighsModel:
    """
    Creates a HiGHS model with the given name, time limit, and logfile.
    :param name: name of the HiGHS model.
    :param time_limit: time limit in seconds for the HiGHS model.
    :param logfile: path to logfile.
    :param node_file: path to node file.
    :return: HiGHS model.
    """
    # Create model
    model = hp.Highs()
    model.setOptionValue("time_limit", time_limit)
    model.setOptionValue("output_flag", False)  # Disable/enable console output

    # Clear the logfile and start logging
    if logfile != "":
        with open(logfile, "w") as _:
            pass
        model.setOptionValue("log_file", logfile)

    return model

def terminal_groups_without_root(connected_components: List[ConnectedComponents], current_cc: ConnectedComponents) -> List[ConnectedComponents]:
    """
    Get terminal groups until index k without kth root.
    :param connected_components: list of ConnectedComponents objects.
    :param current_cc: current connected_component.
    :return: subset of terminals index k to K.
    """
    return [t for cc in connected_components for t in cc.terminals if cc.id >= current_cc.id
            and t != current_cc.root and cc.pipe == current_cc.pipe]


def add_directed_constraints(model: hp.HighsModel, apr: AutomatedPipeRouting) -> Tuple[hp.HighsModel, List[hp.HighsVarType]]:
    """
    Adds DO-D constraints to the model.
    :param model: HiGHS model.
    :param apr: AutomatedPipeRouting-object.
    :return: HiGHS model with DO-D constraints and decision variables.
    """
    # Sets
    k_indices = [(k, l) for k in apr.terminal_list for l in apr.terminal_list if l >= k]

    # Decision variables
    x = {(p.id, e): model.addVariable(0, 1, name=f"x[{p.id},{e}]") for p in apr.pipes for e in apr.graph.edges()}
    y1 = {(p.id, a): model.addVariable(0, 1, name=f"y1[{p.id},{a}]") for p in apr.pipes for a in apr.arcs}
    y2 = {(cc.id, cc.pipe.id, a): model.addVariable(0, 1, name=f"y2[{cc.id},{cc.pipe.id},{a}]") for cc in apr.connected_components
          for a in cc.arcs}
    z = {(k, l): model.addVariable(0, 1, name=f"z[{k},{l}]") for k, l in k_indices}
    b = {v: model.addVariable(0, 1, name=f"b[{v}]") for v in apr.nodes}
    b_dummy_out = {v: model.addVariable(0, 1, name=f"b[{v}]") for v in apr.nodes}

    for col in range(model.getNumCol()):
        model.changeColIntegrality(col, hp.HighsVarType.kInteger)

    # Constraint 1: connection between y2 and y1
    for cc in apr.connected_components:
        for a in cc.arcs:
            model.addConstr(y2[cc.id, cc.pipe.id, a] <= y1[cc.pipe.id, a])

    # Constraint 2: indegree of each vertex cannot exceed 1
    for cc in apr.connected_components:
        for v in apr.nodes:
            model.addConstr(sum(y1[cc.pipe.id, (u, w)] for u, w in cc.arcs if v == w) <= 1)

    # Constraint 3: connection between y1 and x
    for p in apr.pipes:
        for u, v in apr.edges:
            model.addConstr(y1[p.id, (u, v)] + y1[p.id, (v, u)] <= x[p.id, (u, v)])

    # Constraint 4: enforce terminal group rooted at one root
    for cc_k in apr.connected_components:
        model.addConstr(sum(z[cc_l.id, cc_k.id] for cc_l in apr.connected_components if cc_l.id <= cc_k.id and cc_k.pipe == cc_l.pipe) == 1)

    # Constraint 5: enforce one root per arborescence
    for cc_k in apr.connected_components:
        for cc_l in apr.connected_components:
            if cc_l.id > cc_k.id:
                model.addConstr(z[cc_k.id, cc_k.id] >= z[cc_k.id, cc_l.id])

    # Constraint 6: terminals in T^{1···k−1} cannot attach to root r k
    for cc_k in apr.connected_components:
        for t in [cc for cc in apr.connected_components if cc.id < cc_k.id and cc.pipe == cc_k.pipe]:
            LHS = sum(y2[cc_k.id, cc_k.pipe.id, a] for a in cc_k.arcs if a[1] == t)
            if LHS != 0:
                model.addConstr(LHS == 0)

    # Constraint 7: indegree at most outdegree for Steiner points
    for p in apr.pipes:
        for v in apr.steiner_points_per_pipe[p]:
            model.addConstr(sum(y1[p.id, (a[0], a[1])] for a in apr.arcs if a[1] == v) <=
                                sum(y1[p.id, (a[0], a[1])] for a in apr.arcs if a[0] == v))

    # Constraint 8: indegree at most outdegree per terminal group
    for cc in apr.connected_components:
        remaining_vertices = set(apr.nodes) - set(terminal_groups_without_root(apr.connected_components, cc))
        for v in remaining_vertices:
            model.addConstr(sum(y2[cc.id, cc.pipe.id, (a[0], v)] for a in cc.arcs if a[1] == v) <=
                                sum(y2[cc.id, cc.pipe.id, (v, a[1])] for a in cc.arcs if a[0] == v))

    # Constraint 9: connect y2 and z
    for cc_k in apr.connected_components:
        for cc_l in apr.connected_components:
            if cc_l.id > cc_k.id:
                model.addConstr(sum(y2[cc_k.id, cc_k.pipe.id, a] for a in cc_k.arcs if a[1] == cc_l.root) <= z[cc_k.id, cc_l.id])

    # Constraint 10: you cannot use pipe B for connected component A
    for cc_k in apr.connected_components:
        for cc_l in apr.connected_components:
            if cc_k.pipe != cc_l.pipe and cc_l.id >= cc_k.id:
                model.addConstr(z[cc_k.id, cc_l.id] == 0)

    # Constraint 11: prevent crossing of pipes > you can only enter a vertex with one pipe type
    for v in apr.nodes:
        for p1 in apr.pipes:
            LHS = sum(y1[p1.id, a] for a in apr.arcs if a[1] == v)
            for p2 in apr.pipes:
                if p1 != p2:
                    RHS = sum(y1[p2.id, a] for a in apr.arcs if a[1] == v)
                    model.addConstr(1 - LHS >= RHS)

    # Constraint 12: detects bends where they occur
    for v in apr.nodes:
        for p in apr.pipes:
            for n1 in apr.graph.neighbors(v):
                for n2 in apr.graph.neighbors(v):
                    if np.sum(np.array(np.array(n1) != np.array(n2))) >= 2:
                        model.addConstr(y1[p.id, (n1, v)] + y1[p.id, (v, n2)] - 1 <= b[v])
                        model.addConstr(y1[p.id, (n2, v)] + y1[p.id, (v, n1)] - 1 <= b[v])
                        # pass

    return model, x, y1, y2, z, b, b_dummy_out


def demand_and_supply_directed(apr: AutomatedPipeRouting, cc_k: ConnectedComponents, t: Tuple[int, int, int],
                               v: Tuple[int, int, int], z: hp.HighsVarType) -> Union[hp.HighsVarType, int]:
    """
    Calculate the demand and supply for a directed model.
    :param cc_k: The current connected component.
    :param t: A terminal represented as a tuple of integers.
    :param v: A vertex represented as a tuple of integers.
    :param z: The decision variable z.
    :return: The value of z if the vertex is the root, -z if the vertex is a terminal, and 0 otherwise.
    """

    # We assume terminals are disjoint from each other! LATER NOG NAAR KIJKEN!
    cc_l = [cc for cc in apr.connected_components if t in cc.terminals][0]

    if v == cc_k.root:
        return z[(cc_k.id, cc_l.id)]
    elif v == t:
        return -z[(cc_k.id, cc_l.id)]
    else:
        return 0


def add_flow_constraints(model: hp.HighsModel, apr: AutomatedPipeRouting, z: hp.HighsVarType, y2: hp.HighsVarType) -> Tuple[hp.HighsModel, List[hp.HighsVarType]]:
    """
    We add the flow constraints to the Highs model.
    :param model: Highs model.
    :param apr: AutomatedPipeRouting-object.
    :param z: decision variable z.
    :param y2: decision variable y2.
    :return: Highs model and variable(s).
    """
    # Decision variables (binary flow variables)
    f = {(cc.id, t, cc.pipe.id, a): model.addVariable(0, 1, hp.HighsVarType.kInteger, name=f"f[{cc.id},{cc.pipe.id},{a}]") for cc in apr.connected_components
          for t in terminal_groups_without_root(apr.connected_components, cc) for a in cc.arcs}

    # Constraint 1: flow conservation
    for v in apr.nodes:
        for cc in apr.connected_components:
            for t in terminal_groups_without_root(apr.connected_components, cc):
                first_term = sum(f[cc.id, t, cc.pipe.id, a] for a in cc.arcs if a[0] == v)
                second_term = sum(f[cc.id, t, cc.pipe.id, a] for a in cc.arcs if a[1] == v)
                left_hand_side = first_term - second_term
                demand_and_supply = demand_and_supply_directed(apr, cc, t, v, z)
                model.addConstr(left_hand_side == demand_and_supply)

    # Constraint 2: connection between f and y2
    for cc in apr.connected_components:
        for t in terminal_groups_without_root(apr.connected_components, cc):
            for a in cc.arcs:
                left_hand_side = f[cc.id, t, cc.pipe.id, a]
                right_hand_side = y2[cc.id, cc.pipe.id, a]
                model.addConstr(left_hand_side <= right_hand_side)

    # Constraint 3: prevent flow from leaving a terminal
    for cc in apr.connected_components:
        for t in terminal_groups_without_root(apr.connected_components, cc):
            if sum(1 for u, v in cc.arcs if u == t) > 0:
                left_hand_side = sum(f[cc.id, t, cc.pipe.id, (u, v)] for u, v in cc.arcs if u == t)
                model.addConstr(left_hand_side == 0, name="flow_3")

    return model, f


def build_model(apr: AutomatedPipeRouting, time_limit: float, logfile: str = "") -> Tuple[hp.HighsModel, float]:
    """
    Returns the deterministic directed model.
    :param apr: AutomatedPipeRouting-object.
    :param time_limit: time limit in seconds for the HiGHS model.
    :param logfile: path to logfile.
    :return: HiGHS model.
    """
    # Create the model
    logging.info("Building the model.")

    model = make_model(time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    model, x, y1, y2, z, b, b_dummy_out = add_directed_constraints(model, apr)
    model, f = add_flow_constraints(model, apr, z, y2)

    # End tracking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    logging.info(f"Model built in {compilation_time:.2f} seconds.")

    return model, x, y1, y2, z, f, b, b_dummy_out


def run_model(model: hp.HighsModel, apr: AutomatedPipeRouting, x: hp.HighsVarType, b: hp.HighsVarType) -> Dict[Pipe, List[Tuple[int, int, int]]]:
    """
    Solves the model and returns the result.
    :param model: highspy model.
    :param apr: AutomatedPipeRouting-object.
    :param x: highspy variable.
    :return: dictionary with pipes and list of selected edges.
    """
    logging.info(f"Started with running the model...")

    # Optimize model
    model.minimize(sum(x[p.id, e] * apr.graph.edges[e]["weight"] * p.costs for p in apr.pipes for e in apr.edges) + 0.1 * sum(b[v] for v in apr.nodes))

    logging.info(f"Runtime: {model.getRunTime():.2f} seconds")

    return {p: [e for e in apr.edges if model.variableValue(x[p.id, e]) == 1] for p in apr.pipes}


# def toy_example():
#     """
#     Toy example used to debug and illustrate the model.
#     """
#     pipe = Pipe(1, 1)
#     cc = ConnectedComponents(1, [1, 3], pipe, [(1, 2), (2, 3)])
#     graph = nx.Graph()
#     edges = [(1, 2), (2, 3)]
#     graph.add_edges_from(edges)
#     for edge in graph.edges():
#         graph.edges[edge]["weight"] = 1
#     apr = AutomatedPipeRouting([cc], graph, False)
#
#     model, x, *_ = build_model(apr, 3600)
#     result = run_model(model, apr, x)
#     print(result)
