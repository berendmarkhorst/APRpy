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

    plot_space_and_route(original_search_space, obstacles, result)


def example_dong_and_bian(case_nr: int = 0):
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
    
    cases = {'case1': [(99, 0, 99), (0, 46, 0)], 
             
             'case2': [(0, 0, 99), (99, 46, 0)],
             
             'case3': [(0, 0, 99), (99, 46, 0), #case 2 with 3 pipes
                       (1, 0, 99), (98, 46, 0),
                       (2, 0, 99), (97, 46, 0)],
             
             'case4': [(99, 0, 99), (0, 46, 0), # case 1 with 3 pipes
                       (98, 0, 99), (1, 46, 0),
                       (97, 0, 99), (2, 46, 0)],
             }
    pipe_starts_ends = cases['case'+str(case_nr+1)]
    all_connected_components = []
    pipeid = 1
    for start, end in zip(pipe_starts_ends[0::2], pipe_starts_ends[1::2]):
        pipe_i = Pipe(id=pipeid, costs=1)
        print(start, end)
        connected_components_i = ConnectedComponents(1, [start, end], pipe_i, list(graph.edges()))
        all_connected_components.append(connected_components_i)
        pipeid += 1

    apr = AutomatedPipeRouting(all_connected_components, graph, False)

    apr = step4(apr, 1, 10)

    # apr = simplify_graph(apr)
    model, x, y1, y2, z, f, b = build_model(apr, 3600)
    result = run_model(model, apr, x, b)

    plot_space_and_route(search_space, obstacles, result)

def example_dong_and_bian_equipment_model():
    """
    Example from Dong and Bian used to illustrate the model.
    :return:
    """
    # (-5000,200,-1000),(-2000,2000,-3000),

    # (-1000,-350,-2300),(-500,-250,-3000),(-900,-100,-2500),(-600,-250,-2800),
    # (-1000,250,-2500),(-500,-100,-2950),(-875,-100,-2250),(-625,150,-2500),
    
    # (-1000,-1350,-2300),(-500,-1250,-3000),(-900,-1100,-2500),(-600,-1250,-2800),
    # (-1000,-750,-2500),(-500,-1100,-2950),(-875,-1100,-2250),(-625,-850,-2500),
    
    # (400,-400,-1800),(1800,1400,-3000),
    
    # (2600,-400,-1800),(4000,1400,-3000),
    
    # (-2800,-1800,2700),(-1700,-2000,-900),(-2550,-1300,1350),(-1950,-1800,900),
    # (-2410,-1550,1590),(-2090,-1800,1350),(-2650,-1150,2230),(-1885,-1800,1590),
    # (-2465,-1600,2530),(-1615,-1800,2230),(-1885,-1600,2230),(-1615,-1800,1930),
    
    # (1200,-1800,2700),(2300,-2000,900),(1450,-1300,1350),(2050,-1800,900),
    # (1590,-1550,1590),(1910,-1800,1350),(1375,-1150,2230),(2115,-1800,1590),
    # (1535,-1600,2530),(2385,-1800,2330),(2115,-1600,2230),(2385,-1800,1930),
    
    # (-3600,-2000,1700),(-5000,600,3000),(-4650,800,2550),(-3950,600,2150)
    # (-3900,-2000,1200),(-4670-1300,1700)
    
    # (3300,200,2540),(4600,-2000,1640),(3550,-200,2720),(4350,-2000,2540)
    # (3750,360,2200),(4150,200,1980),(4300,-1500,1240),(3600,-2000,1640)
    
    # (-900-1400,1350),(400,-2000,650),(-600-1850,2690),(100,-2000,1690)
    # (-700,-1670,1690),(200,-2000,1350),(-950,-1150,2690),(350,-1850,1690)
    # (-650,-1790,2910),(50,-1490,2690),(-850,-910,2690),(250,-1150,1690)
    # (-850,-1310,2970),(250,-910,2690),(250,-1310,1410),(-850,-910,1690)

    

if __name__ == "__main__":
    # toy_example()
    example_dong_and_bian(case_nr=1)