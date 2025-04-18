# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:21:28 2025

@author: roywinter
"""

from src.space_modeling import *
from src.objects import *
from src.mathematical_model import *
from src.visualize import *
import numpy as np
import json

# from illustrative_examples import solve_pipe_problem

import numpy as np
from collections import deque

def write_random_case_json(search_space, obstacles, result, seed, fill_ratio, pipe_lengths):
    """

    Parameters
    ----------
    search_space : np.array(x,y,z)
        representation of search space
    obstacles : np.array(n,6)
        n obstacles with lower and upper corners
    result : random pipe routes
        list of tuples where origin is the first tuple and destination is last tuple
    seed : int
        seed integer
    fill_ratio : float
        how much percent is at least filled
    pipe_lengths : list
        integers that represent the pipe length in a list.

    Returns
    -------
    json_case : dictionary
        random generated case description in dictionary format.

    """
    seed_str = '_seed'+str(seed)
    size_str = '_size'+'_'.join(map(str,search_space.shape))
    pipes_str = '_pipes'+'_'.join(map(str,pipe_lengths))
    fill_str = '_fill'+str(fill_ratio)
    fill_str = fill_str.replace('.', '')
    
    json_case = {}
    json_case['name'] = 'randomcase'+seed_str+size_str+pipes_str+fill_str
    json_case['size'] = list(search_space.shape)
    json_case['obstacles'] = obstacles.tolist()
    json_case['pipes'] = []
    pipei = 1
    for pipe in result:
        pipe_casei = {}
        pipe_casei['id'] = pipei
        pipe_casei['diameter'] = 1 
        pipe_casei['costs'] = 1
        pipe_casei['connected_components'] = []
        components = {}
        components['id'] = pipei
        components['terminals'] = [[int(x) for x in result[pipe][0][0]], [int(x) for x in result[pipe][-1][1]]]
        components['forbidden_nodes'] = []
        pipe_casei['connected_components'].append(components)
        json_case['pipes'].append(pipe_casei)
        pipei += 1
    
    
    with open('Instances/random_instances/'+json_case['name']+'.json', 'w') as fp:
        json.dump(json_case, fp, indent=4)
        
    return json_case
    
def is_valid_move(x, y, z, space, pipe_id, shape):
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2] and (space[x, y, z] == 1 or space[x, y, z] == pipe_id)

def bfs_manhattan_distance(space, start, end, pipe_id, memory):
    if space[start] == 0 or space[end] == 0:
        return -1  # No path if start or end is an obstacle
    if (start,end) in memory:
        return memory[(start,end)]
    
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    shape = space.shape
    while queue:
        (x, y, z), dist = queue.popleft()
        
        if (x, y, z) == end:
            return dist
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if is_valid_move(nx, ny, nz, space, pipe_id, shape) and (nx, ny, nz) not in visited:
                visited.add((nx, ny, nz))
                queue.append(((nx, ny, nz), dist + 1))
                memory[(start,(nx, ny, nz))] = dist+1
    
    return -1  # No path found

def is_corner_coordinate(coord, array):
    shape = array.shape
    corners = [
        (0, 0, 0),
        (0, 0, shape[2] - 1),
        (0, shape[1] - 1, 0),
        (0, shape[1] - 1, shape[2] - 1),
        (shape[0] - 1, 0, 0),
        (shape[0] - 1, 0, shape[2] - 1),
        (shape[0] - 1, shape[1] - 1, 0),
        (shape[0] - 1, shape[1] - 1, shape[2] - 1)
    ]
    return coord in corners

def create_random_pipe_route(pipe_length, pipe_space, pipe_id, verbose=False):
    """
    Parameters
    ----------
    pipe_length : integer, length of pipe
    pipe_space : np.array with 1s for unoccupied spaces
    pipe_id : integer, 2 or larger

    Returns
    -------
    pipe_space : np.array with pipe_id where the space is occupied with a pipe
    pipe_route : the random route described in a list of tuples that represent origin and destionation

    """
    previous_move_direction = np.array([0,0,0])
    # Generate a random starting point within the array
    start = tuple(np.random.randint(0, s) for s in pipe_space.shape)
    while pipe_space[start] != 1:
        start = tuple(np.random.randint(0, s) for s in pipe_space.shape)
    
    # Initialize the pipe route with start coordinate
    pipe_route = [(start,start)]
    
    # Switch the boolean at the starting point
    pipe_space[start] = pipe_id
    current_distance = 0
    
    # Define possible moves (6 directions in 3D space)
    moves = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    
    length_so_far = 0
    bfs_memory = {}
    
    while length_so_far < pipe_length:
        current_point = np.array(pipe_route[-1][1])
        
        # Generate a random move
        move_direction = moves[np.random.randint(0, len(moves))]
        # make sure that new move is not the previous move in the opposite direction
        while all(move_direction == previous_move_direction*-1):
            move_direction = moves[np.random.randint(0, len(moves))] 
        
        #define move size that does not go out of the space or is longer than intended pipe length
        move_size = np.random.randint(1, pipe_space.shape)
        move_size = np.minimum(move_size, np.array([pipe_length-length_so_far, pipe_length-length_so_far, pipe_length-length_so_far]))
        move = move_direction * move_size
        
        # Calculate the new destination point
        new_point = current_point + move
        pipe_part = np.hstack((current_point, new_point))
        pipe_part = fix_obstacle_ordering(pipe_part)

        # Check if the new point is within bounds, check if 1 coordinate is already visited, and manhatten distance (incooperating obstacles) is larger than previous point.
        if (all(0 <= new_point[i] < pipe_space.shape[i] for i in range(3)) and 
            np.all(np.logical_or( pipe_space[pipe_part[0]:pipe_part[3]+1, pipe_part[1]:pipe_part[4]+1, pipe_part[2]:pipe_part[5]+1]==1 , 
                          pipe_space[pipe_part[0]:pipe_part[3]+1, pipe_part[1]:pipe_part[4]+1, pipe_part[2]:pipe_part[5]+1]==pipe_id))):
            new_distance = bfs_manhattan_distance(pipe_space, start, tuple(new_point), pipe_id, bfs_memory)
            if new_distance > current_distance:  
                if verbose:
                    print('Pipe found so far:',new_distance, 'Pipe length goal', pipe_length)
                previous_move_direction = move_direction
                pipe_route.append((tuple(current_point), tuple(new_point)))
                pipe_space[pipe_part[0]:pipe_part[3]+1, pipe_part[1]:pipe_part[4]+1, pipe_part[2]:pipe_part[5]+1] = pipe_id
                length_so_far += abs(sum(move))
                current_distance = new_distance
                #if the new location is in the corner there is no logic next move anymore as this will not increase manhatten distance.
                if is_corner_coordinate(tuple(new_point), pipe_space):
                    if verbose:
                        print('With the random walk you reached the corner and we can no longer increase the pipe length without making irrational moves. The goal pipe length was: ', pipe_length, 'a random pipe with a pipe length of', length_so_far,' is proposed. You can try with a different seed number to change the starting coordinate')
                    return pipe_space, pipe_route[1:]
    
    return pipe_space, pipe_route[1:]

def random_pipe_obstacle_problem_gemerator(fill_percentage, search_size, pipe_lengths,verbose=False):
    """
    creates one or more single pipe routes from random chosen location with a user defined pipe length and problem space size
    Starts with creating a random pipe and then random obstacles that do not collide with pipe.
    The spaces are occupied with at least the fill_percentage

    Parameters
    ----------
    fill_percentage : float < 1
        floating point percentage that describes how full the space is with obstacles+pipes
    search_size : list or set with 3 integers in there
    pipe_lengths : list of integers of pipe lengths where each integer > 1

    Returns
    -------
    search space : filled with obstacles = 0, pipes with their pipe id, and 1 for free space
    obstacles : list of liststs with obstacles, one obstacle is represented by lower left corner and upper right corner
    result_random: list of tuples that represent the pipe route from coorindate to coordinate.
    """
    sizex, sizey, sizez, = search_size
    search_space = np.ones((sizex, sizey, sizez))
    pipe_space = search_space.copy()
    if verbose:
        print('Start creating random pipes with lengths', pipe_lengths, 'and fill the space for a percentage of',100*fill_percentage)
    result_random = {}
    for pipe_i in range(len(pipe_lengths)):
        pipe_space, pipe_route = create_random_pipe_route(pipe_lengths[pipe_i], pipe_space, pipe_i+2, verbose)
        result_random['Pipe ' + str(pipe_i+1)] = pipe_route
        if verbose:
            print('Finished Pipe',pipe_i+1)
    
    if verbose:
        plot_space_and_route(pipe_space, np.array([[0,0,0,1,1,1]]), result_random)#, saveTitle='illustrativeExamplesmall2Route')
    
    search_space = np.ones((sizex, sizey, sizez))
    if verbose:
        print('Create Obstacles')
    obstacles = []
    pipe_lengths = np.sum(pipe_space!=1)
    unoccupied_search_space = np.sum(search_space)
    while sizex*sizey*sizez - unoccupied_search_space + pipe_lengths < sizex*sizey*sizez*fill_percentage:
        obstacle = np.random.randint([0,0,0,0,0,0], [sizex+1, sizey+1, sizez+1, sizex+1, sizey+1, sizez+1]) #lower and upper bound        
        obstacle = fix_obstacle_ordering(obstacle) # guarantee that lowerbound is lower than higher bound by switching where nessesary
        # if lower and upper bound of obstacle ar not equal and 
        # if space is not already fully filled by other obstacle and
        # if not any space is already occupied by a pipe
        if not( obstacle[0] == obstacle[3] or obstacle[1] == obstacle[4] or obstacle[2] == obstacle[5]) and \
            not np.any(search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]==0) and \
                not np.any(pipe_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]>=2):
            search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
            unoccupied_search_space = np.sum(search_space)
            obstacles.append(obstacle)
            if verbose:
                print((sizex*sizey*sizez - (np.sum(search_space) - pipe_lengths))/(sizex*sizey*sizez), '% occupied')
    
    obstacles = np.array(obstacles)
    if verbose:
        plot_space_and_route(search_space, obstacles, result_random)#, saveTitle='illustrativeExamplesmall2')
    return search_space, obstacles, result_random

# todo = needs a termination criteria otherwise stays stuck in while loop. 
# def random_obstacle_pipe_problem_gemerator(fill_percentage, search_size, pipe_lengths):
#     """
#     creates a single pipe route from random chosen locations with a user defined pipe length and problem space size
#     Starts with creating obstacles and then creates a random pipe thtat do not collide with obstacles. 

#     Parameters
#     ----------
#     fill_percentage : float < 1
#         floating point percentage that describes how full the space is
#     sizex : int > 1
#         size x that is the length of the problem space
#     sizey : int > 1
#         size y that is the depth of the problem space
#     sizez : int > 1
#         size z that is the height of the problem space
#     pipe_lengths : int > 1
#         total random pipe length. 
#     solve : boolean
#         solve the problem or not with the default algorithm. 

#     Returns
#     -------
#     None.

#     """
#     sizex, sizey, sizez, = search_size
#     search_space = np.ones((sizex, sizey, sizez))
#     pipe_space = search_space.copy()
    
#     obstacles = []
#     while sizex*sizey*sizez - np.sum(search_space) + sum(pipe_lengths) < sizex*sizey*sizez*fill_percentage:
#         obstacle = np.random.randint([0,0,0,0,0,0], [sizex+1, sizey+1, sizez+1, sizex+1, sizey+1, sizez+1]) #lower and upper bound        
#         obstacle = fix_obstacle_ordering(obstacle) # guarantee that lowerbound is lower than higher bound by switching where nessesary
#         # if lower and upper bound ar not equal and 
#         # if space is not already fully filled by other obstacle and
#         # if not any space is already occupied by a pipe
#         if not( obstacle[0] == obstacle[3] or obstacle[1] == obstacle[4] or obstacle[2] == obstacle[5]) and \
#             not np.any(search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]==0) and \
#                 not np.any(pipe_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]]>=2):
#             search_space[obstacle[0]:obstacle[3], obstacle[1]:obstacle[4], obstacle[2]:obstacle[5]] = 0
#             obstacles.append(obstacle)
    
#     result_random = {}
#     for pipe_i in range(len(pipe_lengths)):
#         pipe_space, pipe_route = create_random_pipe_route(pipe_lengths[pipe_i], search_space, pipe_i+2)
#         result_random['Pipe ' + str(pipe_i+1)] = pipe_route
            
#     obstacles = np.array(obstacles)

#     obstacles = np.array(obstacles)
#     plot_space_and_route(search_space, obstacles, result_random)
#     return search_space, obstacles, result_random


if __name__ == "__main__":
    for seed in range(50):
        # seed = np.random.randint(1000) #118 984 862 #42 #26 #899 #266
        pipe_lengths = [150]
        search_size = [100,100,100]
        fill_ratio = 0.3
        np.random.seed(seed)
        search_space, obstacles, result = random_pipe_obstacle_problem_gemerator(fill_ratio, search_size, pipe_lengths,verbose=False)
        json_case = write_random_case_json(search_space, obstacles, result, seed, fill_ratio, pipe_lengths)
        print(seed)
    for seed in range(50):    
        pipe_lengths = [200]
        search_size = [100,100,100]
        fill_ratio = 0.3
        np.random.seed(seed)
        search_space, obstacles, result = random_pipe_obstacle_problem_gemerator(fill_ratio, search_size, pipe_lengths,verbose=False)
        json_case = write_random_case_json(search_space, obstacles, result, seed, fill_ratio, pipe_lengths)
        print(seed)
    for seed in range(50):    
        pipe_lengths = [250]
        search_size = [100,100,100]
        fill_ratio = 0.3
        np.random.seed(seed)
        search_space, obstacles, result = random_pipe_obstacle_problem_gemerator(fill_ratio, search_size, pipe_lengths,verbose=False)
        json_case = write_random_case_json(search_space, obstacles, result, seed, fill_ratio, pipe_lengths)
        print(seed)
    for seed in range(50):    
        pipe_lengths = [300]
        search_size = [100,100,100]
        fill_ratio = 0.3
        np.random.seed(seed)
        search_space, obstacles, result = random_pipe_obstacle_problem_gemerator(fill_ratio, search_size, pipe_lengths,verbose=False)
        json_case = write_random_case_json(search_space, obstacles, result, seed, fill_ratio, pipe_lengths)
        print(seed)

    # todo:
    """hang the obstacles to the pipe. """        
    """branches"""
    """towers/cylinders, sparce matrices, diameter pipe"""