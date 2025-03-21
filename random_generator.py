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
    random_obstacle_pipe_problem_gemerator(0.5, 100, 100, 100, [200], solve=True)
    
    random_pipe_obstacle_problem_gemerator(0.2, 100, 100, 100, [500], solve=True)
    