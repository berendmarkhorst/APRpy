import numpy as np
from scipy.ndimage import convolve
import networkx
import logging
import time

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

def step3(voxelarray) -> networkx.Graph:
    """
    Create a graph from a 3D binary array.
    :param voxelarray: 3D binary array.
    :return: networkx graph.
    """
    logging.info("Building the graph.")

    start = time.time()

    graph = networkx.Graph()

    # Find indices of all True elements
    true_indices = np.argwhere(voxelarray)

    # Convert to list of tuples
    nodes = [tuple(map(int, index)) for index in true_indices]

    # Add nodes to the graph
    graph.add_nodes_from(nodes)

    # Add edges
    for node in graph.nodes:
        x, y, z = node

        neighbors = [(x-1, y, z), (x+1, y, z), (x, y-1, z), (x, y+1, z), (x, y, z-1), (x, y, z+1)]

        for neighbor in neighbors:
            if neighbor in graph.nodes:
                graph.add_edge(node, neighbor)
                weight = np.linalg.norm(np.array(node) - np.array(neighbor))
                graph[node][neighbor]['weight'] = weight

    duration = time.time() - start

    logging.info(f"Graph built in {duration:.2f} seconds with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")


    return graph
