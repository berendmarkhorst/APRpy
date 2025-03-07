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


def step1(voxelarray):
    """
    Basic space modeling: remove voxels with 6 neighbors.
    :param voxelarray: 3D binary array representing a room.
    :return: 3D binary array.
    """
    # Perform the convolution to count neighbors
    neighbor_count = convolve(voxelarray.astype(int), kernel, mode='constant', cval=0)

    # Set elements with 6 True neighbors to False
    voxelarray[(neighbor_count == 6) & (voxelarray == True)] = False

    return voxelarray


def step2(voxelarray):
    # Perform the convolution to count neighbors
    neighbor_count = convolve(voxelarray.astype(int), kernel, mode='constant', cval=0)

    # Identify edge elements
    edge_mask = np.zeros_like(voxelarray, dtype=bool)
    edge_mask[0, :, :] = edge_mask[-1, :, :] = True
    edge_mask[:, 0, :] = edge_mask[:, -1, :] = True
    edge_mask[:, :, 0] = edge_mask[:, :, -1] = True

    # Set elements with 5 True neighbors to True if they're not at the edges
    voxelarray[(neighbor_count == 5) & ~edge_mask] = True

    # Set elements with 4 True neighbors to True if they're at the edges
    voxelarray[(neighbor_count == 4) & edge_mask] = True

    # Set the remaining elements to False
    voxelarray[~((neighbor_count == 5) & ~edge_mask) & ~((neighbor_count == 4) & edge_mask)] = False

    return voxelarray


def step3(voxelarray, original_voxelarray) -> nx.Graph:
    """
    Create a graph from a 3D binary array using a vectorized approach.
    Nodes are the True voxels and edges connect the nearest True voxel
    in the positive x, y, and z directions if all intermediate voxels in
    original_voxelarray are True.
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
                    weight = x2 - x1  # Unit steps, so difference is the Euclidean distance.
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
                    weight = y2 - y1
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
                    weight = z2 - z1
                    graph.add_edge(node1, node2, weight=weight)

    duration = time.time() - start
    logging.info(f"Graph built in {duration:.2f} seconds with {len(graph.nodes)} nodes and {len(graph.edges)} edges (vectorized).")
    return graph


# def step3(voxelarray, original_voxelarray) -> networkx.Graph:
#     """
#     Create a graph from a 3D binary array.
#     :param voxelarray: 3D binary array.
#     :return: networkx graph.
#     """
#     logging.info("Building the graph.")
#
#     start = time.time()
#
#     graph = networkx.Graph()
#
#     # Find indices of all True elements
#     true_indices = np.argwhere(voxelarray)
#
#     # Convert to list of tuples
#     nodes = [tuple(map(int, index)) for index in true_indices]
#
#     # Add nodes to the graph
#     graph.add_nodes_from(nodes)
#
#     # Define the six cardinal directions as (dx, dy, dz)
#     directions = [(1, 0, 0), (-1, 0, 0),
#                   (0, 1, 0), (0, -1, 0),
#                   (0, 0, 1), (0, 0, -1)]
#
#     # Optionally, convert graph.nodes to a list for repeated scanning
#     nodes = list(graph.nodes)
#
#     for node in nodes:
#         x, y, z = node
#         for dx, dy, dz in directions:
#             # Filter candidates that lie in the given direction.
#             if dx != 0:
#                 # Looking along the x-axis: nodes must share y and z.
#                 candidates = [n for n in nodes if
#                               n[1] == y and n[2] == z and ((dx > 0 and n[0] > x) or (dx < 0 and n[0] < x))]
#                 # Use the difference in x-coordinate as the distance.
#                 if candidates:
#                     neighbor = min(candidates, key=lambda n: abs(n[0] - x))
#             elif dy != 0:
#                 # Looking along the y-axis: nodes must share x and z.
#                 candidates = [n for n in nodes if
#                               n[0] == x and n[2] == z and ((dy > 0 and n[1] > y) or (dy < 0 and n[1] < y))]
#                 if candidates:
#                     neighbor = min(candidates, key=lambda n: abs(n[1] - y))
#             elif dz != 0:
#                 # Looking along the z-axis: nodes must share x and y.
#                 candidates = [n for n in nodes if
#                               n[0] == x and n[1] == y and ((dz > 0 and n[2] > z) or (dz < 0 and n[2] < z))]
#                 if candidates:
#                     neighbor = min(candidates, key=lambda n: abs(n[2] - z))
#
#             # If a valid neighbor is found in this direction, add an edge.
#             if candidates:
#                 intermediate_steps = get_intermediate_steps(node, neighbor)
#                 if np.sum(original_voxelarray[intermediate_steps]) == len(intermediate_steps):
#                     graph.add_edge(node, neighbor)
#                     weight = np.linalg.norm(np.array(neighbor) - np.array(node))
#                     graph[node][neighbor]['weight'] = weight
#
#     duration = time.time() - start
#
#     logging.info(f"Graph built in {duration:.2f} seconds with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
#
#     return graph

def get_intermediate_steps(node, neighbor):
    """
    Given two nodes that differ in exactly one coordinate,
    return a list of intermediate steps (excluding the start and end)
    with a step size of 1.
    """
    steps = []
    # Calculate difference vector
    diff = (neighbor[0] - node[0], neighbor[1] - node[1], neighbor[2] - node[2])

    if diff[0] != 0:
        step = 1 if diff[0] > 0 else -1
        # Exclude the start and end by using range from node[0]+step to neighbor[0]
        for x in range(node[0] + step, neighbor[0], step):
            steps.append((x, node[1], node[2]))
    elif diff[1] != 0:
        step = 1 if diff[1] > 0 else -1
        for y in range(node[1] + step, neighbor[1], step):
            steps.append((node[0], y, node[2]))
    elif diff[2] != 0:
        step = 1 if diff[2] > 0 else -1
        for z in range(node[2] + step, neighbor[2], step):
            steps.append((node[0], node[1], z))

    return steps

def step4(apr: AutomatedPipeRouting, k: float, bend_penalty: float):
    """
    Reduce the graph heuristically.
    :param apr: AutomatedPipeRouting object.
    :param k: number of paths we consider.
    :return:
    """
    start = time.time()

    original_graph = apr.graph.copy()

    nodes_to_be_kept = set(t for cc in apr.connected_components for t in cc.terminals)

    for cc in apr.connected_components:
        for t1 in cc.terminals:
            for t2 in cc.terminals:
                if t1 != t2:
                    for _ in range(k):
                        # path = a_star_minimize_bends(apr.graph, t1, t2, bend_penalty=bend_penalty)
                        path = nx.astar_path(apr.graph, t1, t2, weight='weight')

                        # Make edges from path
                        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                        for e in edges:
                            apr.graph.edges[e]['weight'] *= 100

                        nodes_to_be_kept.update(path)

        # Reset the weights
        for edge in apr.graph.edges:
            apr.graph.edges[edge]['weight'] = manhattan_distance1(edge[0], edge[1])

    remove_nodes = set(apr.graph.nodes) - nodes_to_be_kept

    # Remove nodes from apr.graph that are not in nodes
    apr.graph.remove_nodes_from(remove_nodes)
    apr.nodes = list(apr.graph.nodes)
    apr.edges = list(apr.graph.edges)
    apr.arcs = list(apr.graph.edges) + [(v, u) for (u, v) in apr.graph.edges]

    # Per pipe type, collect the steiner points
    _, _, apr.steiner_points_per_pipe = apr.add_connected_components(apr.connected_components)

    for cc in apr.connected_components:
        cc.edges = [edge for edge in apr.edges if edge[0] in cc.allowed_nodes and edge[1] in cc.allowed_nodes]
        cc.arcs = cc.edges + [(v, u) for u, v in cc.edges]

    duration = time.time() - start
    logging.info(f"Reduced graph from {len(original_graph.nodes)} to {len(apr.graph.nodes)} nodes in {duration:.2f} seconds.")
    logging.info(f"Reduced graph from {len(original_graph.edges)} to {len(apr.graph.edges)} edges in {duration:.2f} seconds.")

    return apr


def manhattan_distance2(graph, p, q):
    return graph[p][q]['weight']

def manhattan_distance1(p, q):
    return int(np.sum(np.abs(np.array(p) - np.array(q))))


def a_star_minimize_bends(graph, start, goal, bend_penalty=10):
    # The state is a tuple: (current_node, previous_node)
    # For start, previous is None
    open_set = []
    heappush(open_set, (0, start, None))

    # Dictionaries for path reconstruction and cost tracking
    came_from = {}  # key: (node, prev), value: (previous state)
    cost_so_far = {(start, None): 0}

    while open_set:
        current_priority, current, prev = heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = [current]
            state = (current, prev)
            while state in came_from:
                state = came_from[state]
                path.append(state[0])
            path.reverse()
            return path

        for neighbor in graph.neighbors(current):
            # Compute the distance cost from current to neighbor
            dist = manhattan_distance2(graph, current, neighbor)
            new_cost = cost_so_far[(current, prev)] + dist

            # If there's a previous node, check for a bend
            if prev is not None:
                # Vectors: from prev->current and current->neighbor
                vec1 = (current[0] - prev[0], current[1] - prev[1], current[2] - prev[2])
                vec2 = (neighbor[0] - current[0], neighbor[1] - current[1], neighbor[2] - current[2])
                norm1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)
                norm2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)
                if norm1 > 0 and norm2 > 0:
                    # Normalize vectors
                    dir1 = (vec1[0] / norm1, vec1[1] / norm1, vec1[2] / norm1)
                    dir2 = (vec2[0] / norm2, vec2[1] / norm2, vec2[2] / norm2)
                    # If the directions differ (you could allow a tolerance for near collinearity)
                    if dir1 != dir2:
                        new_cost += bend_penalty

            new_state = (neighbor, current)
            if new_state not in cost_so_far or new_cost < cost_so_far[new_state]:
                cost_so_far[new_state] = new_cost
                priority = new_cost + manhattan_distance1(neighbor, goal)
                heappush(open_set, (priority, neighbor, current))
                came_from[new_state] = (current, prev)

    # No path found
    return None


def simplify_graph(apr):
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
                            H.edges[u, v]['weight'] = manhattan_distance1(u, v)
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

def bresenham_3d(start, end):
    """
    Generate 3D Bresenham line coordinates between start and end points.

    :param start: Tuple[int, int, int], starting coordinate (x1, y1, z1)
    :param end: Tuple[int, int, int], ending coordinate (x2, y2, z2)
    :return: List[Tuple[int, int, int]], list of coordinates on the line
    """
    x1, y1, z1 = start
    x2, y2, z2 = end
    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            points.append((x1, y1, z1))
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            points.append((x1, y1, z1))
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            points.append((x1, y1, z1))
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    points.append((x1, y1, z1))
    return points

