import json
import numpy as np
from .objects import AutomatedPipeRouting, Pipe, ConnectedComponents
from .space_modeling import remove_cells_with_six_neighbors, create_graph_from_voxel_array, reduce_graph_heuristically, remove_degree_two_nodes
import time


def parse_apr_from_json(json_path: str) -> AutomatedPipeRouting:
    """
    Parses a JSON-formatted benchmark file for automated pipe routing.

    This function reads an APR benchmark instance from a JSON file and constructs the corresponding
    `AutomatedPipeRouting` object. The JSON file should include the size of the search space, obstacles,
    and one or more pipe objects with their associated terminal components and costs.

    The following assumptions are made:
    - Obstacles are defined as 6-tuples marking blocked rectangular prisms: [x1, y1, z1, x2, y2, z2]
    - Pipes may have multiple connected components (sets of terminals)
    - All terminals are assumed to be within bounds and not inside obstacles
    - `remove_cells_with_six_neighbors` and `create_graph_from_voxel_array` are responsible for space modeling and graph creation respectively

    :param json_path: Path to the JSON file
    :return: An instance of `AutomatedPipeRouting`
    """

    # Start measuring the time
    start_time = time.time()

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize the 3D search space
    size = data['size']
    search_space = np.ones((size, size, size), dtype=int)

    # Add obstacles to the 3D grid
    obstacles = []
    for obstacle in data['obstacles']:
        x1, y1, z1, x2, y2, z2 = obstacle
        search_space[x1:x2+1, y1:y2+1, z1:z2+1] = 0
        obstacles.append(obstacle)

    original_search_space = search_space.copy()

    # Gather all terminals across all pipes for initial space modeling
    all_terminals = [
        tuple(t)
        for pipe_data in data['pipes']
        for cc_data in pipe_data['connected_components']
        for t in cc_data['terminals']
    ]

    search_space = remove_cells_with_six_neighbors(search_space, all_terminals)

    # Convert the binary search space to a graph
    graph = create_graph_from_voxel_array(search_space, original_search_space)

    # Create a list to hold all connected components
    all_connected_components = []

    for pipe_data in data['pipes']:
        pipe = Pipe(id=pipe_data['id'], costs=pipe_data["costs"])

        for cc_data in pipe_data['connected_components']:
            terminals = [tuple(t) for t in cc_data['terminals']]
            cc_id = cc_data['id']

            # Construct allowed edges
            forbidden_nodes = cc_data["forbidden_nodes"]
            original_graph = graph.copy()
            original_graph.remove_nodes_from(forbidden_nodes)

            allowed_edges = list(original_graph.edges())

            cc = ConnectedComponents(
                id=cc_id,
                terminals=terminals,
                pipe=pipe,
                edges=allowed_edges
            )

            all_connected_components.append(cc)

    # Create the AutomatedPipeRouting object
    apr = AutomatedPipeRouting(
        connected_components=all_connected_components,
        graph=graph,
        name=data["name"]
    )

    # Store the original search space for later use with plotting
    apr.search_space = original_search_space
    apr.obstacles = np.array(obstacles)

    # Simplify the graph even more
    apr = reduce_graph_heuristically(apr)
    apr = remove_degree_two_nodes(apr)

    # Stop measuring the time and attach it to the apr object.
    end_time = time.time()
    elapsed_time = end_time - start_time
    apr.build_instance_time = elapsed_time

    return apr

