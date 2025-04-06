import networkx as nx
import numpy as np
from scipy.ndimage import convolve
import networkx
import logging
import time
import itertools
from .objects import *
from heapq import heappush, heappop
import math
import copy

# Define the 3D kernel to count neighbors
kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                   [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])


def remove_cells_with_six_neighbors(voxelarray, terminals):
    """
    Remove voxels with 6 neighbors, except for terminals.

    This function performs the following steps:
    1. Counts the number of neighbors for each voxel using a 3D convolution.
    2. Sets voxels with exactly 6 neighbors to False, indicating they are removed.
    3. Ensures that terminal voxels remain True, regardless of their neighbor count.

    :param voxelarray: 3D binary array representing a room.
    :param terminals: List of terminal coordinates as tuples (x, y, z).
    :return: Modified 3D binary array with specified voxels removed.
    """
    # Perform the convolution to count neighbors
    neighbor_count = convolve(voxelarray.astype(int), kernel, mode='constant', cval=0)

    # Set elements with 6 True neighbors to False
    voxelarray[(neighbor_count == 6) & (voxelarray == True)] = False

    # Set elements in terminals to True in a vectorized way
    if len(terminals) > 0:
        # Convert list of tuples to three arrays (x, y, z)
        x, y, z = zip(*terminals)

        # Use advanced indexing to set values to True
        voxelarray[x, y, z] = True

    return voxelarray


def create_graph_from_voxel_array(voxelarray, original_voxelarray) -> nx.Graph:
    """
    Create a graph from a 3D binary array using a vectorized approach.

    This function performs the following steps:
    1. Adds nodes for all True voxels in the voxel array.
    2. Adds edges between neighboring True voxels in the positive x, y, and z directions,
       ensuring that all intermediate voxels in the original voxel array are True.

    :param voxelarray: 3D binary array representing the voxel space.
    :param original_voxelarray: 3D binary array representing the original voxel space.
    :return: NetworkX graph with nodes and edges representing the voxel connectivity.
    """
    logging.info("Building the graph (vectorized).")
    start = time.time()
    nx_dim, ny, nz = voxelarray.shape
    graph = nx.Graph()

    # Add nodes: all indices where voxelarray is True.
    true_indices = np.argwhere(voxelarray)
    nodes = [tuple(map(int, idx)) for idx in true_indices]
    graph.add_nodes_from(nodes)

    # Process positive x-direction: loop over all (y, z) slices.
    for y in range(ny):
        for z in range(nz):
            # Find all x indices where the voxel is True.
            x_coords = np.nonzero(voxelarray[:, y, z])[0]
            if len(x_coords) > 1:
                # Since x_coords is sorted, the immediate neighbor is the next element.
                for i in range(len(x_coords) - 1):
                    x1, x2 = x_coords[i], x_coords[i+1]
                    # If there is a gap, check that all intermediate voxels are True.
                    if x2 > x1 + 1 and not np.all(original_voxelarray[x1+1:x2, y, z]):
                        continue
                    node1 = (int(x1), int(y), int(z))
                    node2 = (int(x2), int(y), int(z))
                    weight = int(x2 - x1)  # Unit steps, so difference is the Euclidean distance.
                    graph.add_edge(node1, node2, weight=weight)

    # Process positive y-direction: loop over all (x, z) slices.
    for x in range(nx_dim):
        for z in range(nz):
            y_coords = np.nonzero(voxelarray[x, :, z])[0]
            if len(y_coords) > 1:
                for i in range(len(y_coords) - 1):
                    y1, y2 = y_coords[i], y_coords[i+1]
                    if y2 > y1 + 1 and not np.all(original_voxelarray[x, y1+1:y2, z]):
                        continue
                    node1 = (int(x), int(y1), int(z))
                    node2 = (int(x), int(y2), int(z))
                    weight = int(y2 - y1)
                    graph.add_edge(node1, node2, weight=weight)

    # Process positive z-direction: loop over all (x, y) slices.
    for x in range(nx_dim):
        for y in range(ny):
            z_coords = np.nonzero(voxelarray[x, y, :])[0]
            if len(z_coords) > 1:
                for i in range(len(z_coords) - 1):
                    z1, z2 = z_coords[i], z_coords[i+1]
                    if z2 > z1 + 1 and not np.all(original_voxelarray[x, y, z1+1:z2]):
                        continue
                    node1 = (int(x), int(y), int(z1))
                    node2 = (int(x), int(y), int(z2))
                    weight = int(z2 - z1)
                    graph.add_edge(node1, node2, weight=weight)

    duration = time.time() - start
    logging.info(f"Graph built in {duration:.2f} seconds with {len(graph.nodes)} nodes and {len(graph.edges)} edges (vectorized).")
    return graph

def reduce_graph_heuristically(apr: AutomatedPipeRouting):
    """
    Reduce the graph heuristically by keeping only necessary nodes and edges.

    This function performs the following steps:
    1. Copies the original graph and initializes sets for terminal nodes and nodes to be kept.
    2. Sorts the connected components by pipe ID to prevent overlapping solutions with different pipes.
    3. Iterates through connected components, finding paths between terminals and updating nodes to be kept.
    4. Removes nodes that are not in the set of nodes to be kept.
    5. Updates the APR object with the reduced graph and recalculates Steiner points per pipe.

    :param apr: AutomatedPipeRouting object.
    :return: Modified AutomatedPipeRouting object with a reduced graph.
    """
    start = time.time()

    original_graph = apr.graph.copy()
    original_nr_nodes = apr.graph.number_of_nodes()
    original_nr_edges = apr.graph.number_of_edges()

    terminal_set = set(t for cc in apr.connected_components for t in cc.terminals)
    nodes_to_be_kept = terminal_set

    # Sort the APR connected components by the pipe ID to prevent overlapping solutions with different pipes.
    apr.connected_components = sorted(apr.connected_components, key=lambda cc: cc.pipe.id)

    current_pipe = apr.connected_components[0].pipe
    nodes_to_be_kept_current_pipe = set()

    for cc in apr.connected_components:
        if cc.pipe != current_pipe:
            current_pipe = cc.pipe
            remove_nodes = nodes_to_be_kept_current_pipe - terminal_set
            apr.graph.remove_nodes_from(remove_nodes)
            nodes_to_be_kept_current_pipe = set()

        for t1 in cc.terminals:
            for t2 in cc.terminals:
                if t1 != t2:
                    path = nx.astar_path(apr.graph, t1, t2, weight='weight')
                    nodes_to_be_kept_current_pipe.update(path)
                    nodes_to_be_kept.update(path)

    apr.graph = original_graph
    remove_nodes = set(apr.graph.nodes) - nodes_to_be_kept
    apr.graph.remove_nodes_from(remove_nodes)
    apr.nodes = list(apr.graph.nodes)
    apr.edges = list(apr.graph.edges)
    apr.arcs = list(apr.graph.edges) + [(v, u) for (u, v) in apr.graph.edges]

    _, _, apr.steiner_points_per_pipe = apr.add_connected_components(apr.connected_components)

    for cc in apr.connected_components:
        cc.edges = [edge for edge in apr.edges if edge[0] in cc.allowed_nodes and edge[1] in cc.allowed_nodes]
        cc.arcs = cc.edges + [(v, u) for u, v in cc.edges]

    duration = time.time() - start
    logging.info(f"Reduced graph from {original_nr_nodes} to {len(apr.graph.nodes)} nodes in {duration:.2f} seconds.")
    logging.info(f"Reduced graph from {original_nr_edges} to {len(apr.graph.edges)} edges in {duration:.2f} seconds.")

    return apr


def manhattan_distance(graph, p, q):
    """
    Calculate the Manhattan distance between two nodes in the graph.

    This function performs the following steps:
    1. If the graph is provided, return the weight of the edge between nodes p and q.
    2. If the graph is not provided, calculate the Manhattan distance between nodes p and q.

    :param graph: NetworkX graph object or None.
    :param p: First node as a tuple (x, y, z).
    :param q: Second node as a tuple (x, y, z).
    :return: Manhattan distance between nodes p and q.
    """
    if graph is not None:
        return graph[p][q]['weight']
    else:
        return int(np.sum(np.abs(np.array(p) - np.array(q))))


def remove_degree_two_nodes(apr: AutomatedPipeRouting):
    """
    Remove nodes with degree two from the graph, except for terminal nodes.

    This function performs the following steps:
    1. Copies the original graph to avoid modifying it directly.
    2. Identifies nodes to be preserved, including terminal nodes and nodes with a degree other than two.
    3. Iteratively removes nodes with degree two that are not in the preserved set, merging their neighbors.
    4. Updates the APR object with the reduced graph and recalculates Steiner points per pipe.

    :param apr: AutomatedPipeRouting object.
    :return: Modified AutomatedPipeRouting object with a reduced graph.
    """
    start = time.time()

    G = apr.graph
    # Make a copy so we don't modify the original graph.
    H = G.copy()
    # Mark the nodes to be preserved (based on the original graph)
    preserved = {n for n in G.nodes() if G.degree(n) != 2} | {t for cc in apr.connected_components for t in cc.terminals}
    # (Optional) Handle the case where no node is preserved:
    if not preserved and len(G) > 0:
        # Arbitrarily keep one node (for instance, the first)
        preserved = {next(iter(G.nodes()))}

    # Iteratively remove nodes that are not preserved and have degree 2.
    # We use a loop because removing a node can make another non-preserved node’s degree become 2.
    changed = True
    while changed:
        changed = False
        # Iterate over a static list of nodes (since we will modify H)
        for node in list(H.nodes()):
            # Only remove nodes that are not in the original preserved set.
            if node not in preserved and H.degree(node) == 2:
                # Get the two neighbors
                neighbors = list(H.neighbors(node))
                if len(neighbors) == 2:
                    u, v = neighbors

                    if np.sum(np.array(u) != np.array(v)) == 1:
                        # (Optional) If your edges have data (like weights), you could merge them here.
                        # For now, we simply add an edge between u and v if it doesn't already exist.
                        if not H.has_edge(u, v):
                            H.add_edge(u, v)
                            H.edges[u, v]['weight'] = manhattan_distance(None, u, v)
                        # Remove the degree-2 node.
                        H.remove_node(node)
                        changed = True

    apr.graph = H
    apr.nodes = list(apr.graph.nodes)
    apr.edges = list(apr.graph.edges)
    apr.arcs = list(apr.graph.edges()) + [(v, u) for (u, v) in apr.graph.edges()]

    # Per pipe type, collect the steiner points
    _, _, apr.steiner_points_per_pipe = apr.add_connected_components(apr.connected_components)

    for cc in apr.connected_components:
        cc.edges = [edge for edge in apr.edges if edge[0] in cc.allowed_nodes and edge[1] in cc.allowed_nodes]
        cc.arcs = cc.edges + [(v, u) for u, v in cc.edges]

    duration = time.time() - start

    logging.info(f"Reduced graph from {len(G.nodes)} to {len(apr.graph.nodes)} nodes in {duration:.2f} seconds.")
    logging.info(f"Reduced graph from {len(G.edges)} to {len(apr.graph.edges)} edges in {duration:.2f} seconds.")

    return apr
