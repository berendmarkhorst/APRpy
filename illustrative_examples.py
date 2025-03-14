from src.space_modeling import *
from src.objects import *
from src.mathematical_model import *
from src.visualize import *
import numpy as np



def fix_obstacle_ordering(obstacle):
    """
    obstacle with coordinates x, y ,z, x2, y2, z2
    x y z should all be smaller then or equal to x2 y2 z2
    This function fixes the order so that is guaranteed. 
    
    Parameters
    ----------
    obstacle : np array of size 6

    Returns
    -------
    obstacle : np.array of size 6

    """
    obstacles_l = max(obstacle[0], obstacle[3])
    obstacles_s = min(obstacle[0], obstacle[3])  
    obstacle[0] = obstacles_s
    obstacle[3] = obstacles_l
    
    obstacles_l = max(obstacle[1], obstacle[4])
    obstacles_s = min(obstacle[1], obstacle[4]) 
    obstacle[1] = obstacles_s
    obstacle[4] = obstacles_l
    
    obstacles_l = max(obstacle[2], obstacle[5])
    obstacles_s = min(obstacle[2], obstacle[5]) 
    obstacle[2] = obstacles_s
    obstacle[5] = obstacles_l
    return obstacle


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
                         [16,1,26,40,40,30]])

    for obstacle in obstacles:
        search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
    
    original_search_space = search_space.copy()
       
    # Apply space modeling
    search_space = step1(search_space)
       
    # Convert binary 3D array to graph
    graph = step3(search_space, original_search_space)
    
    #todo: direction of endpoints = X,X?
    cases = {'case1': [(20,1,1), (20,40,40)], # table 2
             
             'case2': [(20, 1, 1), (20, 40, 30), 
                       (20, 1, 30), (20, 40, 1)], #table3 multiple pipes
             
             'case3': [(20, 1, 1), (40, 20, 30), (20, 40, 20), (1, 40, 40)], #table4 branch piping

             'case4': [(20, 1, 1), (40, 20, 30), (20, 40, 20), (1, 40, 40), #table 5, branch piping
                       (20,1,30),(40,20,1)] #table5 the single Pipe

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
       
    # apr = AutomatedPipeRouting(all_connected_components, graph, False) # 20,1,1 is not in graph?
       
    # apr = step4(apr, 1, 10)
       
    # apr = simplify_graph(apr)
    # model, x, y1, y2, z, f, b = build_model(apr, 3600)
    # result = run_model(model, apr, x, b)
       
    plot_space_and_route(search_space, obstacles, {})

def example_dong_and_bian(case_nr: int = 0):
    """
    Example from Dong and Bian used to illustrate the model.
    https://ieeexplore.ieee.org/abstract/document/9172005
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
        'P2': {'connection': [[60,48,5], [81,40,12]], 'diameter':[48,48], 'type': 'single'},
        'P3': {'connection': [[60,48,5], [81,40,12]], 'diameter':[48,48], 'type': 'single'},
        'P4': {'connection': [[88,40,12], [107,34,10]], 'diameter':[48,48], 'type': 'single'},
        'P5': {'connection': [[88,20,12], [151,34,10]], 'diameter':[48,48], 'type': 'single'},
        'P6': {'connection': [[118,31,19], [170,31,19], [22,1,88], [186,1,88]], 'diameter':[64,64,46,46], 'type': 'unequalbranchpipe'},
        'P7': {'connection': [[124,31,19], [164,31,19], [63,11,100], [143,11,100]], 'diameter':[62,62,44,44], 'type': 'unequalbranchpipe'},
        'P8': {'connection': [[130,31,19], [158,31,19], [107,6,105]], 'diameter':[44,44,44], 'type': 'branchpipe'}
        }

    plot_space_and_route(search_space, obstacles, {})


def create_random_pipe_route(pipe_length, pipe_space):
    """
    Parameters
    ----------
    pipe_length : integer, length of pipe
    pipe_space : np.array with 1s for unoccupied spaces

    Returns
    -------
    pipe_space : np.array with 0s where there are pipes
    pipe_route : the random route described in a list of tuples that represent origin and destionation

    """
    
    # Generate a random starting point within the array
    start = tuple(np.random.randint(0, s) for s in pipe_space.shape)
    while pipe_space[start] != 1:
        start = tuple(np.random.randint(0, s) for s in pipe_space.shape)
    
    # Initialize the pipe route with start coordinate
    pipe_route = [(start,start)]
    
    # Switch the boolean at the starting point
    pipe_space[start] = 0
    
    # Define possible moves (6 directions in 3D space)
    moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    length_so_far = 0
    while length_so_far < pipe_length:
        current_point = pipe_route[-1][1]
        
        # Generate a random move
        move = moves[np.random.randint(0, len(moves))]

        move_size = np.random.randint(1, pipe_space.shape)
        move_size = np.minimum(move_size, np.array([pipe_length-length_so_far, pipe_length-length_so_far, pipe_length-length_so_far]))
        move = move * move_size
        
        # Calculate the new point
        new_point = tuple(current_point[i] + move[i] for i in range(3))
        pipe_part = np.array(sum([current_point,new_point], ()))
        pipe_part = fix_obstacle_ordering(pipe_part)

        # Check if the new point is within bounds and not already visited
        if all(0 <= new_point[i] < pipe_space.shape[i] for i in range(3)) and np.sum(pipe_space[pipe_part[0]:pipe_part[3]+1, pipe_part[1]:pipe_part[4]+1, pipe_part[2]:pipe_part[5]+1]==0)==1:
            pipe_route.append((current_point, new_point))
            pipe_space[pipe_part[0]:pipe_part[3]+1, pipe_part[1]:pipe_part[4]+1, pipe_part[2]:pipe_part[5]+1] = 0
            length_so_far += abs(sum(move))
    
    return pipe_space, pipe_route[1:]

def random_pipe_obstacle_problem_gemerator(fill_percentage, sizex, sizey, sizez, pipe_lengths, solve=False):
    """
    creates a single pipe route from random chosen locations with a user defined pipe length and problem space size
    starts with creating a random pipe and then random obstacles that do not collide with pipe.

    Parameters
    ----------
    fill_percentage : float < 1
        floating point percentage that describes how full the space is
    sizex : int > 1
        size x that is the length of the problem space
    sizey : int > 1
        size y that is the depth of the problem space
    sizez : int > 1
        size z that is the height of the problem space
    pipe_lengths : int > 1
        total random pipe length. 
    solve : boolean
        solve the problem or not with the default algorithm. 

    Returns
    -------
    None.

    """
    
    search_space = np.ones((sizex, sizey, sizez))
    pipe_space = search_space.copy()
    
    result_random = {}
    for pipe_i in range(len(pipe_lengths)):
        pipe_space, pipe_route = create_random_pipe_route(pipe_lengths[pipe_i], pipe_space)
        result_random['Pipe ' + str(pipe_i+1)] = pipe_route
        
    search_space = np.ones((sizex, sizey, sizez))
    
    obstacles = []
    while sizex*sizey*sizez - np.sum(search_space) + np.sum(pipe_space==0) < sizex*sizey*sizez*fill_percentage:
        obstacle = np.random.randint([0,0,0,0,0,0], [sizex+1, sizey+1, sizez+1, sizex+1, sizey+1, sizez+1]) #lower and upper bound        
        obstacle = fix_obstacle_ordering(obstacle) # guarantee that lowerbound is lower than higher bound by switching where nessesary
        # if lower and upper bound ar not equal and 
        # if space is not already fully filled by other obstacle and
        # if not any space is already occupied by a pipe
        if not( obstacle[0] == obstacle[3] or obstacle[1] == obstacle[4] or obstacle[2] == obstacle[5]) and \
            not np.all(search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]==0) and \
                not np.any(pipe_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]==0):
            search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
            obstacles.append(obstacle)
    
    obstacles = np.array(obstacles)
    plot_space_and_route(search_space, obstacles, result_random)

    if solve:
        result_arp = solve_pipe_problem(search_space, obstacles, result_random)
        plot_space_and_route(search_space, obstacles, result_arp)

def random_obstacle_pipe_problem_gemerator(fill_percentage, sizex, sizey, sizez, pipe_lengths, solve=False):
    """
    creates a single pipe route from random chosen locations with a user defined pipe length and problem space size
    Starts with creating obstacles and then creates a random pipe thtat do not collide with obstacles. 

    Parameters
    ----------
    fill_percentage : float < 1
        floating point percentage that describes how full the space is
    sizex : int > 1
        size x that is the length of the problem space
    sizey : int > 1
        size y that is the depth of the problem space
    sizez : int > 1
        size z that is the height of the problem space
    pipe_lengths : int > 1
        total random pipe length. 
    solve : boolean
        solve the problem or not with the default algorithm. 

    Returns
    -------
    None.

    """
    
    search_space = np.ones((sizex, sizey, sizez))
    pipe_space = search_space.copy()
    
    obstacles = []
    while sizex*sizey*sizez - np.sum(search_space) + sum(pipe_lengths) < sizex*sizey*sizez*fill_percentage:
        obstacle = np.random.randint([0,0,0,0,0,0], [sizex+1, sizey+1, sizez+1, sizex+1, sizey+1, sizez+1]) #lower and upper bound        
        obstacle = fix_obstacle_ordering(obstacle) # guarantee that lowerbound is lower than higher bound by switching where nessesary
        # if lower and upper bound ar not equal and 
        # if space is not already fully filled by other obstacle and
        # if not any space is already occupied by a pipe
        if not( obstacle[0] == obstacle[3] or obstacle[1] == obstacle[4] or obstacle[2] == obstacle[5]) and \
            not np.all(search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]==0):
            search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
            obstacles.append(obstacle)
    
    result_random = {}
    for pipe_i in range(len(pipe_lengths)):
        pipe_space, pipe_route = create_random_pipe_route(pipe_lengths[pipe_i], search_space)
        result_random['Pipe ' + str(pipe_i+1)] = pipe_route
            
    obstacles = np.array(obstacles)
    plot_space_and_route(search_space, obstacles, result_random)

    if solve:
        result_arp = solve_pipe_problem(search_space, obstacles, result_random)
        plot_space_and_route(search_space, obstacles, result_arp)

if __name__ == "__main__":
    # toy_example()
    # example_jiang_etall()
    # example_dong_and_bian(case_nr=1)
    # example_dong_and_bian_equipment_model()
    
    random_obstacle_pipe_problem_gemerator(0.2, 100, 100, 100, [200], solve=True)
    
    # random_pipe_obstacle_problem_gemerator(0.2, 10, 10, 10, [20], solve=True)
    