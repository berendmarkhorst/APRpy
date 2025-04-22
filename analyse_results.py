# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:17:18 2025

@author: roywinter
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

results = {}
for root, subdirs, files in os.walk("Instances/Random_instances_1/"):
    if root != 'Instances/Random_instances_1/':
        results[root] = {'filled_percentages': [], 'nr_obstacles': [],  'nr_bends': [], 'lengths': [], 'total_times': []}
    
    for file in files:
        # if root in experiment_list:
        if file[-4:] == '.pkl':
            pklfile = root+'/'+file
            with open(pklfile, 'rb') as fp:
                data = pickle.load(fp)
            
            total = np.product(data.search_space.shape)
            occupied = np.sum(data.search_space==0)
            percentage_filled = occupied / total
            results[root]['filled_percentages'].append(percentage_filled)
            
            nr_obstacles = len(data.obstacles)
            results[root]['nr_obstacles'].append(nr_obstacles)
            
            total_time = data.build_instance_time + data.mathematical_model_time
            results[root]['total_times'].append(total_time)
            
            bends = data.nr_bends
            results[root]['nr_bends'].append(bends)
                
            length = data.length
            results[root]['lengths'].append(length)


aggregated = {}
for experiment in results:
    aggregated[experiment] = {}
    for subResult in results[experiment]:
        aggregated[experiment][subResult] = [np.mean(results[experiment][subResult]), 
                                             np.std(results[experiment][subResult]), 
                                             np.median(results[experiment][subResult]), 
                                             np.min(results[experiment][subResult]), 
                                             np.max(results[experiment][subResult])]


fig = plt.figure()
experiment_pipes = [50, 100, 150, 200, 250, 300]
positions = range(1, len(experiment_pipes) + 1)  # Positions for box plots

for i, experiment_pipe in enumerate(experiment_pipes):
    experiment_folder = 'Instances/Random_instances_1/experiments 100100100 ' + str(experiment_pipe) + ' 03'
    plt.boxplot(results[experiment_folder]['total_times'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [f'{pipe}' for pipe in experiment_pipes])
plt.xlabel('Pipe Length')
plt.ylabel('Seconds required to solve')
plt.title('Box Plot of time required find different path lengths \n Search space 100x100x100 with 30% obstacle ratio')

fig.savefig('time_vs_Pipe_Length.pdf', bbox_inches='tight')



fig = plt.figure()
fillratios = ['01','02','03','04','05']
positions = range(1, len(fillratios) + 1)  # Positions for box plots

for i, fill_ratio in enumerate(fillratios):
    experiment_folder = 'Instances/Random_instances_1/experiments 100100100 200150 ' + fill_ratio
    plt.boxplot(results[experiment_folder]['total_times'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [str(int(ratio)*10)+'%' for ratio in fillratios])
plt.xlabel('Obstacle Ratio')
plt.ylabel('Seconds required to solve')
plt.title('Box Plot of time required given different Obstacle Ratios \n Search space 100x100x100 with pipe lengths 200 and 150.')

fig.savefig('time_vs_obstacle_ratio.pdf', bbox_inches='tight')


