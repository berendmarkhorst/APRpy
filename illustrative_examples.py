from src.space_modeling import *
from src.objects import *
from src.mathematical_model import *
from src.visualize import *
import numpy as np


def toy_example():
    """
    Toy example used to debug and illustrate the model.
    """
    # Make the 3D binary array
    size = 10
    search_space = np.ones((size, size, size))

    # Make an easy set of obstacles
    obstacles = np.array([[2, 2, 2, 4, 4, 4], [6, 6, 6, 8, 8, 8]])

    for obstacle in obstacles:
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0

    original_search_space = search_space.copy()

    # Apply space modeling
    search_space_reduced = step1(search_space)

    # Convert binary 3D array to graph
    graph = step3(search_space_reduced, original_search_space)

    # Make small toy example
    pipe1 = Pipe(1, 1)
    pipe2 = Pipe(2, 1)
    connected_components1 = ConnectedComponents(1, [(0, 0, 0), (0, 9, 9), (2, 2, 1)], pipe1, list(graph.edges()))
    connected_components2 = ConnectedComponents(2, [(9, 0, 0), (9, 9, 9)], pipe2, list(graph.edges()))
    all_connected_components = [connected_components1, connected_components2]
    apr = AutomatedPipeRouting(all_connected_components, graph, False)

    apr = step4(apr, 1, 10)

    apr = simplify_graph(apr)

    model, x, y1, y2, z, f, b = build_model(apr, 3600)
    result = run_model(model, apr, x, b)

    # for v in apr.nodes:
    #     print(f"Node {v}: {model.variableValue(b[v])}")

    # for p in [pipe1]:
    #     for e in apr.edges:
    #         if model.variableValue(x[p.id, e]) > 0:
    #             print(f"Pipe {p.id} edge {e} {model.variableValue(x[p.id, e]):.2f}")

    included_nodes = {p: set() for p in apr.pipes}
    # Loop over the edges and also include all the nodes between the source and sink of the edge
    for pipe in apr.pipes:
        for edge in result[pipe]:
            included_nodes[pipe].update(bresenham_3d(edge[0], edge[1]))

    plot_space_and_route(original_search_space, included_nodes)


def example_dong_and_bian():
    """
    Example from Dong and Bian used to illustrate the model.
    :return:
    """
    # Make the 3D binary array
    size = 100
    search_space = np.ones((size, size, size))

    obstacles = np.array([[-45, -50, 0, -30, 20, 20],
                          [0, -50, 10, 50, 10, 30],
                          [0, -50, -30, 50, 50, -10],
                          [-40, -50, 30, -20, 50, 50],
                          [-50, -50, -30, -20, 50, -20],
                          [-20, -10, -40, 0, 0, 20],
                          [-50, -50, -20, -30, -20, 0],
                          [-30, -50, -40, 10, 30, -50],
                          [-20, -50, -40, 0, -40, 20],
                          [-10, 0, -10, 0, 50, 10],
                          [30, -50, 40, 20, -40, 50],
                          [-10, -50, 20, -20, -10, 50],
                          [-14, -10, -16, -6, -16, 8]]) + 50

    for obstacle in obstacles:
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0

    original_search_space = search_space.copy()

    # Apply space modeling
    search_space = step1(search_space)

    # Convert binary 3D array to graph
    graph = step3(search_space, original_search_space)

    # Make small toy example
    pipe1 = Pipe(1, 1)
    pipe2 = Pipe(2, 1)
    connected_components1 = ConnectedComponents(1, [(99, 0, 99), (0, 46, 0)], pipe1, list(graph.edges()))
    # connected_components2 = ConnectedComponents(2, [(1, 0, 0), (1, 1, 0)], pipe2, list(graph.edges()))
    # all_connected_components = [connected_components1, connected_components2]
    all_connected_components = [connected_components1]
    apr = AutomatedPipeRouting(all_connected_components, graph, False)

    apr = step4(apr, 1, 10)

    # apr = simplify_graph(apr)
    model, x, y1, y2, z, f = build_model(apr, 3600)
    result = run_model(model, apr, x)

    plot_space_and_route(search_space, result)


if __name__ == "__main__":
    toy_example()
    # example_dong_and_bian()