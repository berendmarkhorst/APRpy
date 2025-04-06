from src.space_modeling import *
from src.objects import *
from src.mathematical_model import *
from src.visualize import *
from src.parser import parse_apr_from_json
import numpy as np
import pickle


def solve_pipe_problem(search_space, obstacles, result):
    original_search_space = search_space.copy()

    search_space = step1(search_space)
    
    # Convert binary 3D array to graph
    graph = step3(search_space, original_search_space)
    
    cases = {}
    i = 1
    for pipe in result:
        cases['case'+str(i)] = [result[pipe][0][0], result[pipe][-1][-1]]
        i += 1    
    
    case_nr=0
    pipe_starts_ends = cases['case'+str(case_nr+1)]
    all_connected_components = []
    pipeid = 1
    for start, end in zip(pipe_starts_ends[0::2], pipe_starts_ends[1::2]):
        pipe_i = Pipe(id=pipeid, costs=1)
        connected_components_i = ConnectedComponents(1, [start, end], pipe_i, list(graph.edges()))
        all_connected_components.append(connected_components_i)
        pipeid += 1

    apr = AutomatedPipeRouting(all_connected_components, graph, False)
   
    apr = step4(apr, 1, 10)
   
    # apr = simplify_graph(apr)
    model, x, y1, y2, z, f, b = build_model(apr, 3600)
    result = run_model(model, apr, x, b)
    return result

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

def example_jiang_etall(case_nr: int = 0):
    """
    example from W.-Y. Jiang et al. / Ocean Engineering 
    https://www.sciencedirect.com/science/article/pii/S0029801815001031?via%3Dihub
    :return:
    """
    
    size= 40
    search_space = np.ones((size,size,size)) #todo: do a check, this space goes from 1 to 40....
    
    obstacles = np.array([[6,12,1,24,16,40], # table 1
                         [32,12,1,40,16,40],
                         [1,26,1,8,30,40],
                         [16,26,1,34,30,40],
                         [1,1,12,24,40,16],
                         [16,1,26,40,40,30]]) - 1 # @ Roy: deze truc hebben we net besproken!

    for obstacle in obstacles:
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
    
    #todo: direction of endpoints = X,X?
    cases = {'case1': [[(20,1,1), (20,40,40)]], # table 2

             'case2': [[(20, 1, 1), (20, 40, 30)],
                       [(20, 1, 30), (20, 40, 1)]], #table3 multiple pipes

             'case3': [(20, 1, 1), (40, 20, 30), (20, 40, 20), (1, 40, 40)], #table4 branch piping

             'case4': [[(20, 1, 1), (40, 20, 30), (20, 40, 20), (1, 40, 40), #table 5, branch piping
                       (20,1,30), (40,20,1)]] #table5 the single Pipe

             }
    
    pipe_starts_ends = cases['case'+str(case_nr+1)]
    pipe_starts_ends = [(x-1, y-1 , z-1) for components in pipe_starts_ends for x, y, z in components]

    original_search_space = search_space.copy()

    # Apply space modeling
    search_space = step1(search_space, pipe_starts_ends)

    # Convert binary 3D array to graph
    graph = step3(search_space, original_search_space)
    all_connected_components = []
    pipeid = 1
    for start, end in zip(pipe_starts_ends[0::2], pipe_starts_ends[1::2]):
        pipe_i = Pipe(id=pipeid, costs=1)
        print(start, end)
        connected_components_i = ConnectedComponents(1, [start, end], pipe_i, list(graph.edges()))
        all_connected_components.append(connected_components_i)
        pipeid += 1
       
    apr = AutomatedPipeRouting(all_connected_components, graph, False)
       
    apr = step4(apr)
    apr = simplify_graph(apr)
    breakpoint()
    model, x, y1, y2, z, f, b = build_model(apr, 3600)
    result = run_model(model, apr, x, b)
       
    plot_space_and_route(search_space, obstacles, result)

    
def example_1_min_ruy_park():
    """
    Illustrative example from section 6 10 cuboid obstacles....
    Obstacle 4 was outside of the space, so the last digit of 75 in z coordinate was ignored. 
    https://doi.org/10.1016/j.oceaneng.2022.111789
    Returns
    """    
    search_space = np.ones((100, 100, 30))
    
    obstacles = np.array([[0,0,0,0,0,0],
                            [32,12,0,32,12,0],
                            [57,12,0,57,12,0],
                            [57,27,7,57,27,7],
                            [17,42,0,17,42,0],
                            [57,47,0,57,47,0],
                            [0,77,0,0,77,0],
                            [32,72,0,32,72,0],
                            [67,62,0,67,62,0],
                            [52,62,7,52,62,7]])
    
    size = np.array([[33,38,13],
                     [25,25,23],
                     [30,10,13],
                     [30,5,12],
                     [20,25,28],
                     [20,10,28],
                     [23,20,13],
                     [30,20,18],
                     [20,25,30],
                     [15,5,5]])
    
    obstacles[:,3:] += size

    # preference_space = np.array([5,45,5,10,50,10]) # not well defined
    
    # np.hstack()
    
    cases = { 'case1': [(50, 0, 5), (5,45,5), (70,  95, 5)]
              }
    
    result = {}
    plot_space_and_route(search_space, obstacles, result)

def example_2_min_ruy_park():
    """
    Illustrative example from section 6 with 4 cuboid obstacles and 4 cylindrical obstacles....
    https://doi.org/10.1016/j.oceaneng.2022.111789
    Returns
    """   
    search_space = np.ones((100, 100, 40))
    
    obstacles =np.array([[60,20,0,60,20,0],
                         [20,35,0,20,35,0],
                         [75,50,0,75,50,0],
                         [75,0,0,75,0,0]])
    
    size = np.array([[40,30,40],
                     [40,50,40],
                     [5,20,10],
                     [5,20,10]])
    
    obstacles[:,3:] += size
    
    for obstacle in obstacles:
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
                
    #1 tower = [center_x, center_y, start_z, radius, height_z]
    towers = np.array([[35,10,0,10,20],
                       [8,17,0,8,35],
                       [7,82,0,6,35],
                       [88,83,0,11,25]])
    
    for tower in towers:
        x, y, z = np.indices(search_space.shape)
        center_x, center_y, start_z, radius, height_z = tower
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = (distance <= radius) & (z >= start_z) & (z <= start_z + height_z)
        search_space[mask] = 0
    # preference_space = np.array([5,45,5,10,50,10]) # not well defined
    
    # np.hstack()
    
    cases = { 'case1': [(89, 7, 5), (36,92,2), (92, 57, 2)]
              }
    
    result = {}
    # scatterplot_quick_check(search_space)
    plot_space_and_route(search_space, obstacles, result, towers)
    
    

def example_dong_bian_zhao():
    """
    Illustrative example from section 4.2 with 13 cuboid obstacles....
    https://doi.org/10.1016/j.oceaneng.2022.111789
    Returns
    """    
    size = 100
    search_space = np.ones((size, size, size))
    
    obstacles = np.array([[-45,-50,20,-30,20,0],
                            [0,-50,30,50,10,10],
                            [0,-50,-10,50,50,-30],
                            [-40,-50,50,-20,50,30],
                            [-50,-50,-20,-20,50,-30],
                            [-20,-10,-40,0,0,20],
                            [-50,-50,-20,-30,-20,0],
                            [-30,-50,-50,10,30,-40],
                            [-20,-50,-40,0,-40,20],
                            [-10,0,-10,0,50,10],
                            [30,-50,50,20,-40,40],
                            [-10,-50,50,-20,-10,20],
                            [-14,-10,-16,-6,-16,8]])+50
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
    
    cases = { 'case1': [(99, 0, 99), (0,  46, 0)], 
              'case2': [(99, 0, 99), (99, 46, 0)],
              'case3': [(0,  0, 99), (0,  46, 0)],
              'case4': [(0,  0, 99), (99, 46, 0)]
              }
    
    result = {}
    plot_space_and_route(search_space, obstacles, result)


def example_dong_and_bian(case_nr: int = 0):
    """
    Example from Dong and Bian used to illustrate the model.
    https://ieeexplore.ieee.org/abstract/document/9172005
    
    Illustrative example from section 4.2 with 13 cuboid obstacles....
    https://doi.org/10.1016/j.oceaneng.2022.111789
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
    real world example of equipment room from Dong and Bian with different pipe types....
    https://ieeexplore.ieee.org/abstract/document/9172005
    :return:
    """
    search_space = np.ones((200, 80, 120))
    
    obstacles = np.array([
                        [-5000, 200,    -1000, -2000, 2000,  -3000], 
                        
                        [-1000, -350,   -2300,  -500, -250,  -3000], 
                        [-900,  -100,   -2500, -600,  -250, -2800],  
                        [-1000, 250,    -2500, -500,  -100, -2950],  
                        [-875,  -100,   -2250, -625,  150,  -2500], 
                        
                        [-1000, -1350,  -2300, -500,  -1250,  -3000], 
                        [-900,  -1100,  -2500,  -600, -1250, -2800],  
                        [-1000, -750,   -2500,  -500, -1100, -2950],  
                        [-875,  -1100,  -2250,  -625, -850,  -2500], 
                        
                        [400,   -400,   -1800,  1800, 1400,  -3000], 
                        
                        [2600,  -400,   -1800, 4000,  1400, -3000],  
                        
                        [-2800, -1800,  2700,  -1700,  -2000,  -900],  
                        [-2550, -1300,  1350,  -1950,  -1800,  900], 
                        [-2410, -1550,  1590,  -2090,  -1800,  1350],  
                        [-2650, -1150,  2230,  -1885,  -1800,  1590],  
                        [-2465, -1600,  2530,  -1615,  -1800,  2230],  
                        [-1885, -1600,  2230,  -1615,  -1800,  1930], 
                        
                        [1200,  -1800,  2700, 2300,  -2000,  900], 
                        [1450,  -1300,  1350, 2050,  -1800,  900], 
                        [1590,  -1550,  1590, 1910,  -1800,  1350],  
                        [1375,  -1150,  2230, 2115,  -1800,  1590],  
                        [1535,  -1600,  2530, 2385,  -1800,  2330],  
                        [2115,  -1600,  2230, 2385,  -1800,  1930], 
                        
                        [-5000, -2000,  1700,  -3600,  600,  3000],  
                        [-4650, 800,    2550,  -3950,  600,  2150],  
                        [-4670, -2000,  1200,  -3900, -1300, 1700], 
                        
                        [3300,  200,    2540, 4600,  -2000,  1640],  
                        [3550,  -200,   2720,  4350, -2000, 2540], 
                        [3750,  360,    2200, 4150,  200,  1980],  
                        [3600,  -1500,  1240, 4300,  -2000,  1640], 
                        
                        [-900,  -1400,  1350, 400, -2000, 650],  
                        [-600,  -1850,  2690, 100, -2000, 1690], 
                        [-700,  -1670,  1690, 200, -2000, 1350], 
                        [-950,  -1150,  2690, 350, -1850, 1690], 
                        [-650,  -1790,  2910, 50,  -1490,  2690],  
                        [-850,  -910,   2690,  250,  -1150,  1690],  
                        [-850,  -1310,  2970, 250, -910,  2690],  
                        [-850,   -1310,  1410,  250, -910,  1690]])

    obstacles = obstacles + np.array([5000, 2000, 3000, 5000, 2000, 3000])
    obstacles = obstacles / 50 #it is wierd that this results in coordinates with decimal points.... however this is how it is described in the paper.
    obstacles = np.round(obstacles).astype(int) #round so that it become integers.
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)     #make sure that the obstacles are defined as small to larger values
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
    
    
    pipes = {
        'P1': {'connection': [[60,76,4], [112,68,10], [156,68,10]], 'diameter':[60,60,60], 'type': 'branchpipe'},
        'P2': {'connection': [(60,48,5), (81,40,12)], 'diameter':[48,48], 'type': 'single'},
        'P3': {'connection': [(60,48,5), (81,40,12)], 'diameter':[48,48], 'type': 'single'},
        'P4': {'connection': [(88,40,12), (107,34,10)], 'diameter':[48,48], 'type': 'single'},
        'P5': {'connection': [(88,20,12), (151,34,10)], 'diameter':[48,48], 'type': 'single'},
        'P6': {'connection': [[118,31,19], [170,31,19], [22,1,88], [186,1,88]], 'diameter':[64,64,46,46], 'type': 'unequalbranchpipe'},
        'P7': {'connection': [[124,31,19], [164,31,19], [63,11,100], [143,11,100]], 'diameter':[62,62,44,44], 'type': 'unequalbranchpipe'},
        'P8': {'connection': [[130,31,19], [158,31,19], [107,6,105]], 'diameter':[44,44,44], 'type': 'branchpipe'}
        }

    plot_space_and_route(search_space, obstacles, {})

def example_dong_lin():
    """
    Illustrative example from table 1 with 10 cuboid obstacles. almost the same as Dong and Bian only the last obstacle is different and 4 obstacles are not included. 
    https://doi.org/10.3233/ISP-160123
    Returns
    """    
    size = 20 # = 100/5 because search space is made smaller.  
    search_space = np.ones((size, size, size))
    
    obstacles = np.array([[-45,-50,20,-30,20,0],
                            [0,-50,30,50,10,10],
                            [0,-50,-10,50,50,-30],
                            [-40,-50,50,-20,50,30],
                            [-50,-50,-20,-20,50,-30],
                            [-20,-10,-40,0,0,20],
                            [-50,-50,-20,-30,-20,0],
                            [-30,-50,-50,10,30,-40],
                            [-20,-50,-40,0,-40,20],
                            [-10,-30,-10,40,-15,10]])+50
    obstacles = obstacles / 5 # 
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
    
    cases = {'case1': [(15,0,19), (15,10,0)],
             'case2': [(15,1,19), (15,0,19), 
                       (15,0,19), (15,10,0)],#case 2 with 2 pipes
             'case3': [[0,0,19], [0,16,0], [19,14,10], [12,4,0], [0,0,19]] #case 3 with branching
             }

    
    result = {}
    plot_space_and_route(search_space, obstacles, result)

def example_1_yan_yang_lin():
    """
    Illustrative example from section 4.1 with 7 cuboid obstacles.
    https://doi.org/10.1016/j.oceaneng.2024.117961
    Returns
    """   
    size = 50 
    search_space = np.ones((size, size, size))
    
    obstacles = np.array([[6,0,0,16,5,50],
                          [21,0,0,30,5,4],
                          [42,10,0,50,18,12],
                          [42,26,0,50,42,12],
                          [20,42,0,30,50,50],
                          [0,15,0,9,32,20],
                          [14,14,0,34,35,21]])
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
        
    cases = { 'case1': [(0,0,0),(50,50,50)]
            }
    
    result = {}
    plot_space_and_route(search_space, obstacles, result)


def example_1scaled_yan_yang_lin():
    """
    Illustrative example from section 4.1 with 7 cuboid obstacles but now scaled to 100 times larger space.
    Allocating memory for this is problematic...?
    option:
    class Sparse3DArray:
        def __init__(self):
            self.elements = {}
    
        def set_value(self, x, y, z, value):
            self.elements[(x, y, z)] = value
    
        def get_value(self, x, y, z):
            return self.elements.get((x, y, z), False)
    
    # Create a sparse 3D array
    sparse_array = Sparse3DArray()
    
    # Example of setting a value
    sparse_array.set_value(0, 0, 0, True)
    
    # Example of getting a value
    print(sparse_array.get_value(0, 0, 0))  # Output: True
    print(sparse_array.get_value(1, 1, 1))  # Output: False

    https://doi.org/10.1016/j.oceaneng.2024.117961
    Returns
    """   
    size = 50*100
    shape = (size, size, size)
    # search_space = lil_matrix((5000,5000,5000), dtype=bool)
    
    obstacles = np.array([[6,0,0,16,5,50],
                          [21,0,0,30,5,4],
                          [42,10,0,50,18,12],
                          [42,26,0,50,42,12],
                          [20,42,0,30,50,50],
                          [0,15,0,9,32,20],
                          [14,14,0,34,35,21]])
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
    obstacles = obstacles*100
        
    cases = { 'case1': [(0,0,0),(5000,5000,5000)]
            }

    result = {}
    box = {}
    plot_space_and_route(box, obstacles, result)


def example_2_yan_yang_lin():
    """
    Illustrative example from section 4.2 with 6 cuboid obstacles.
    x,y,z axis projection / visualization are rotated in figures in the paper.
    https://doi.org/10.1016/j.oceaneng.2024.117961
    Returns
    """   
    size = 50 
    search_space = np.ones((size, size, size))
    
    obstacles = np.array([[5,1,5,15,50,15],
                          [1,27,29,30,42,44],
                          [1,1,35,21,20,50],
                          [30,5,1,45,40,20],
                          [32,1,25,47,20,40],
                          [32,1,40,47,8,50]])
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
        
    cases = { 'case1': [[2,2,2],[39,12,48],[45,46,15],[10,44,46]], #case 2 Q1 with branching
              'case2': [[2,2,2],[10,23,48],[40,23,30],[35,46,18]] #case 2 Q2 with branching
            }
    
    result = {}
    plot_space_and_route(search_space, obstacles, result)


def example_3_yan_yang_lin():
    """
    Illustrative example from section 4.3 with 16 cuboid obstacles.
    https://doi.org/10.1016/j.oceaneng.2024.117961
    Returns
    """   
    # search_space = np.ones((10000, 12000, 6000))
    
    obstacles = np.array([[3300,4943,1875,3700,2003,3164],
                            [3410,4450,3165,3590,4900,3290],
                            [4332,4511,550,6348,2495,2700],
                            [6348,3490,790,6435,3510,810],
                            [5654,2306,1754,5674,2495,1766],
                            [5175,2050,1475,4925,588,1725],
                            [5900,1862,1400,5750,1732,1550],
                            [6992,3608,1093,7208,3392,1438],
                            [3810,200,1000,4800,1000,1498],
                            [4805,8120,115,5925,7680,539],
                            [3688,9095,131,4038,8895,331],
                            [4333,7468,786,4543,7308,936],
                            [4878,7463,686,5088,7303,836],
                            [3639,9682,1920,6215,9425,2387],
                            [3120,8450,1550,3380,8100,2150],
                            [3900,9490,3510,6589,9070,4020]])
    
    for obstacle in obstacles:
        obstacle = fix_obstacle_ordering(obstacle)
        
    pipes = {
        'P1': {'connection':[(4250, 3500, 730), (3650, 7400, 730)], 'diameter':[60,60], 'type': 'single'},
        'P2': {'connection':[(3500, 4530, 3164), (3250, 2260, 5347)], 'diameter':[30,30], 'type': 'single'},
        'P3': {'connection':[(4305, 1850, 5290), (4305, 700, 1500)], 'diameter':[10,10], 'type': 'single'},
        'P4': {'connection':[(4405, 1850, 5290), (4405, 700, 1500)], 'diameter':[10,10], 'type': 'single'},
        'P5': {'connection':[(4505, 1850, 5290), (4505, 700, 1500)], 'diameter':[10,10], 'type': 'single'},
        'P6': {'connection':[(4605, 1850, 5290), (4605, 700, 1500)], 'diameter':[10,10], 'type': 'single'},
        'P7': {'connection':[[3500, 2416, 1874], [3250, 2410, 5287],[3805, 700, 1450]], 'diameter':[30,30], 'type':'branchpipe'},
        'P8': {'connection':[[5350, 2445, 730], [461, 2050, 1600],[629, 2095, 1350]], 'diameter':[30,30], 'type':'branchpipe'},
        'P9': {'connection':[[4950, 7900, 540], [4250, 9804, 5820],[3850, 11218, 4318], [3430, 10812, 4200]], 'diameter':[30,30], 'type':'branchpipe'},
        'P10': {'connection':[(4804, 7900, 216), (4012, 8894, 216)], 'diameter':[30,30], 'type': 'single'},
        'P11': {'connection':[[4804, 7900, 440], [1750, 3500, 730],[3850, 8894, 230], [4331, 3500, 730]], 'diameter':[30,30], 'type':'branchpipe'},
        'P12': {'connection':[[3850, 9550, 2420], [7000, 5705, 5350],[3620, 9800, 2974], [3850, 9700, 1178]], 'diameter':[30,30], 'type':'branchpipe'},
        'P13': {'connection':[(4220, 9550, 2436), (3450, 6200, 5631)], 'diameter':[30,30], 'type': 'single'},
        'P14': {'connection':[(6007, 9550, 2439), (3450, 6000, 5535)], 'diameter':[30,30], 'type': 'single'},
        'P15': {'connection':[(6007, 9568, 1915), (4105, 700, 1500)], 'diameter':[30,30], 'type': 'single'},
        'P16': {'connection':[(3850, 9550, 1915), (5830, 9550, 300)], 'diameter':[30,30], 'type': 'single'},
        'P17': {'connection':[(4297, 4039, 660), (2427, 4120, 605)], 'diameter':[30,30], 'type': 'single'},
        'P18': {'connection':[(6440, 3500, 800), (7100, 3500, 1443)], 'diameter':[10,10], 'type': 'single'},
        'P19': {'connection':[(4297, 4042, 2446), (2516, 5005, 650)], 'diameter':[10,10], 'type': 'single'},
        'P20': {'connection':[(4297, 4039, 1780), (2520, 4900, 650)], 'diameter':[10,10], 'type': 'single'},
        'P21': {'connection':[(7100, 3500, 1091), (596, 1850, 150)], 'diameter':[10,10], 'type': 'single'},
        'P22': {'connection':[(1210, 1979,950), (5150, 2494, 690)], 'diameter':[20,20], 'type': 'single'},
        'P23': {'connection':[(5664, 2311, 1760), (5633, 750, 300)], 'diameter':[10,10], 'type': 'single'},
        'P24': {'connection':[(3479, 4821, 3260), (4036, 8581, 1173)], 'diameter':[30,30], 'type': 'single'},
        'P25': {'connection':[[3300, 10884, 4180], [2450, 4233, 1574],[4765, 4512, 730]], 'diameter':[30,30], 'type':'branchpipe'},
        'P26': {'connection':[[5501, 7550, 600], [3775, 8950, 333],[6400, 4720, 5450], [1744, 6500, 4031]], 'diameter':[30,30], 'type':'branchpipe'},
        'P27': {'connection':[(8250, 6635, 2350), (765, 6600, 2350)], 'diameter':[30,30], 'type': 'single'},
        'P28': {'connection':[(8250, 6604, 1497), (2153, 6550, 2232)], 'diameter':[30,30], 'type': 'single'},
        'P29': {'connection':[(7100, 7500, 4350), (7174, 4378, 472)], 'diameter':[20,20], 'type': 'single'}
    }
    
    result = {}
    box = {}
    plot_space_and_route(box, obstacles, result)


def main(input_file: str, output_file: str, plot_result: bool = False):
    """
    Main function to parse an Automated Pipe Routing (APR) instance from a JSON file, build and solve the mathematical model,
    and optionally plot the resulting pipe routes.

    This function performs the following steps:
    1. Parses the APR instance from the specified JSON file using the `parse_apr_from_json` function.
    2. Measures the time taken to build the APR instance.
    3. Builds the mathematical model for the APR instance using the `build_model` function.
    4. Runs the mathematical model to find the optimal pipe routes using the `run_model` function.
    5. Measures the time taken to solve the mathematical model.
    6. Stores the results, including the pipe routes, total length of pipes, and number of bends, in the APR object.
    7. Saves the APR object to a pickle file.
    8. Prints the results, including the total time taken, total length of pipes, and number of bends.
    9. Optionally plots the search space, obstacles, and resulting pipe routes if `plot_result` is set to True.

    :param input_file: Path to the input JSON file containing the APR instance.
    :param output_file: Path to the output pickle file where the APR object will be saved.
    :param plot_result: Boolean flag indicating whether to plot the resulting pipe routes. Default is False.
    """
    apr = parse_apr_from_json(input_file)

    # Start measuring the time
    start_time = time.time()

    # Build the model
    model, x, y1, y2, z, f, b = build_model(apr)

    # Run the model
    pipe_route, length, nr_bends = run_model(model, apr, x, b)

    # End measuring the time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    apr.mathematical_model_time = elapsed_time
    apr.pipe_route = pipe_route
    apr.length = length
    apr.nr_bends = nr_bends

    # Store the apr object in a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(apr, f)

    # Print the results
    print(f"Solved in {apr.mathematical_model_time + apr.build_instance_time:.2f} seconds with {apr.length:.0f} units of pipe and {apr.nr_bends:.0f} bends.")

    if plot_result:
        plot_space_and_route(apr.search_space, apr.obstacles, pipe_route)


if __name__ == "__main__":
    # toy_example()
    # example_jiang_etall(case_nr=1)
    # example_dong_and_bian(case_nr=1)
    # example_dong_and_bian_equipment_model()
    # example_1_min_ruy_park()
    # example_2_min_ruy_park()
    # example_dong_lin()
    # example_1_yan_yang_lin()
    # example_1scaled_yan_yang_lin()
    # example_2_yan_yang_lin()
    # example_3_yan_yang_lin()

    input_file = "Instances/Literature/jiang_etall_case4.json"
    output_file = "Instances/Literature/jiang_etall_case4.pkl"
    main(input_file, output_file, plot_result=True)
    