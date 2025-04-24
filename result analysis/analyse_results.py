# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:17:18 2025

@author: roywinter
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from jonckheere_terpstra_test import jonckheere_terpstra_test
from scipy.stats import pearsonr

results = {}
for root, subdirs, files in os.walk("Instances/Random_instances/"):
    if root != 'Instances/Random_instances/':
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
#anova tests
experiment1 = results['Instances/Random_instances/experiments 100100100 50 03']['total_times']
experiment2 = results['Instances/Random_instances/experiments 100100100 100 03']['total_times']
experiment3 = results['Instances/Random_instances/experiments 100100100 150 03']['total_times']
experiment4 = results['Instances/Random_instances/experiments 100100100 200 03']['total_times']
experiment5 = results['Instances/Random_instances/experiments 100100100 250 03']['total_times']
experiment6 = results['Instances/Random_instances/experiments 100100100 300 03']['total_times']

JT_stat, z, p = jonckheere_terpstra_test([experiment2, experiment3, experiment4, experiment5, experiment6])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")

f_statistic, p_value = f_oneway(experiment2, experiment3, experiment4, experiment5, experiment6)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in time measurements across different pipe lengths.")
else:
    print("There is no significant difference in time measurements across different pipe lengths.")


experiment1 = results['Instances/Random_instances/experiments 100100100 200150 01']['total_times']
experiment2 = results['Instances/Random_instances/experiments 100100100 200150 02']['total_times']
experiment3 = results['Instances/Random_instances/experiments 100100100 200150 03']['total_times']
experiment4 = results['Instances/Random_instances/experiments 100100100 200150 04']['total_times']
experiment5 = results['Instances/Random_instances/experiments 100100100 200150 05']['total_times']

f_statistic, p_value = f_oneway(experiment1, experiment2, experiment3, experiment4, experiment5)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

JT_stat, z, p = jonckheere_terpstra_test([experiment1, experiment2, experiment3, experiment4, experiment5])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in time measurements across different obstacle ratios.")
else:
    print("There is no significant difference in time measurements across different obstacle ratios.")


experiment1 = results['Instances/Random_instances/experiments 505050 10063 03']['total_times']
experiment2 = results['Instances/Random_instances/experiments 757575 15094 03']['total_times']
experiment3 = results['Instances/Random_instances/experiments 100100100 200125 03']['total_times']
experiment4 = results['Instances/Random_instances/experiments 125125125 250156 03']['total_times']
experiment5 = results['Instances/Random_instances/experiments 150150150 300188 03']['total_times']
experiment6 = results['Instances/Random_instances/experiments 175175175 350219 03']['total_times']
experiment7 = results['Instances/Random_instances/experiments 200200200 400250 03']['total_times']

f_statistic, p_value = f_oneway(experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

JT_stat, z, p = jonckheere_terpstra_test([experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in time measurements across different search space sizes.")
else:
    print("There is no significant difference in time measurements across different  search space sizes.")





# pipe length experiment
fig = plt.figure(figsize=(6, 4))
experiment_pipes = [50, 100, 150, 200, 250, 300]
positions = range(1, len(experiment_pipes) + 1)  # Positions for box plots

for i, experiment_pipe in enumerate(experiment_pipes):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 ' + str(experiment_pipe) + ' 03'
    plt.boxplot(results[experiment_folder]['total_times'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [f'{pipe} \n ' for pipe in experiment_pipes])
plt.xlabel('Pipe length')
plt.ylabel('Seconds required to solve')
plt.title('Time required given pipe lengths \n Search space $100^3$ with a 30% obstacle ratio.')

fig.savefig('time_vs_Pipe_Length_boxplot.pdf', bbox_inches='tight')


# fig = plt.figure()

# experiment_pipes = [50, 100, 150, 200, 250, 300]
# means = []
# stds = []
# for i, experiment_pipe in enumerate(experiment_pipes):
#     experiment_folder = 'Instances/Random_instances/experiments 100100100 ' + str(experiment_pipe) + ' 03'
#     means.append(aggregated[experiment_folder]['total_times'][0])
#     stds.append(aggregated[experiment_folder]['total_times'][1])
# means = np.array(means)
# stds = np.array(stds)
    
# plt.plot(experiment_pipes, means, label='Mean', marker='o')
# plt.fill_between(experiment_pipes, means - stds, means + stds, color='lightblue', alpha=0.5, label='Std Dev')

# plt.xlabel('Pipe length')
# plt.ylabel('Seconds required to solve')
# plt.title('Time required find different path lengths \n Search space $100^3$ with 30% obstacle ratio.')
# plt.legend()
# plt.show()
# fig.savefig('time_vs_Pipe_Length_linechart.pdf', bbox_inches='tight')





# fill ratio experiment
fig = plt.figure(figsize=(6, 4))
fillratios = ['01','02','03','04','05']
positions = range(1, len(fillratios) + 1)  # Positions for box plots

for i, fill_ratio in enumerate(fillratios):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 200150 ' + fill_ratio
    plt.boxplot(results[experiment_folder]['total_times'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [str(int(ratio)*10)+'%\n ' for ratio in fillratios])
plt.xlabel('Obstacle ratio')
plt.ylabel('Seconds required to solve')
plt.title('Time required given obstacle ratios \n Search space $100^3$ with pipe lengths 200 and 150.')

fig.savefig('time_vs_obstacle_ratio.pdf', bbox_inches='tight')


# fig = plt.figure()
# fillratios = ['01','02','03','04','05']
# means = []
# stds = []
# for i, fill_ratio in enumerate(fillratios):
#     experiment_folder = 'Instances/Random_instances/experiments 100100100 200150 ' + fill_ratio
#     means.append(aggregated[experiment_folder]['total_times'][0])
#     stds.append(aggregated[experiment_folder]['total_times'][1])
# means = np.array(means)
# stds = np.array(stds)
    
# plt.plot(fillratios, means, label='Mean', marker='o')
# plt.fill_between(fillratios, means - stds, means + stds, color='lightblue', alpha=0.5, label='Std Dev')

# plt.xticks(range(5), [str(int(ratio)*10)+'%' for ratio in fillratios])
# plt.xlabel('Obstacle ratio')
# plt.ylabel('Seconds required to solve')
# plt.title('Time required given different obstacle ratios \n Search space $100^3$ with pipe lengths 200 and 150.')
# plt.legend()
# plt.show()
# fig.savefig('time_vs_obstacle_ratio_linechart.pdf', bbox_inches='tight')




# search space size + pipe length experiment
fig = plt.figure(figsize=(6, 4))
spacesizes = ['505050 10063','757575 15094','100100100 200125','125125125 250156','150150150 300188', '175175175 350219', '200200200 400250']
labels = ['$50^3$ \n 100 63','$75^3$ \n 150 94','$100^3$ \n 200 125','$125^3$ \n 250 156','$150^3$ \n 300 188', '$175^3$ \n 350 219', '$200^3$ \n 400 250']
positions = range(1, len(spacesizes) + 1)  # Positions for box plots

for i, spacesize in enumerate(spacesizes):
    experiment_folder = 'Instances/Random_instances/experiments '+spacesize+' 03'
    plt.boxplot(results[experiment_folder]['total_times'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [size for size in labels])
plt.xlabel('Search space size and 2 pipe lengths')
plt.ylabel('Seconds required to solve')
plt.title('Time required given search space and pipe lengths \n Search Space with a 30% obstacle ratio.')
fig.savefig('time_vs_search_space_and_pipe_length.pdf', bbox_inches='tight')


# fig = plt.figure()
# spacesizes = ['505050 10063','757575 15094','100100100 200125','125125125 250156','150150150 300188', '175175175 350219', '200200200 400250']
# labels = ['$50^3$ \n 100 63','$75^3$ \n 150 94','$100^3$ \n 200 125','$125^3$ \n 250 156','$150^3$ \n 300 188', '$175^3$ \n 350 219', '$200^3$ \n 400 250']
# means = []
# stds = []
# for i, spacesize in enumerate(spacesizes):
#     experiment_folder = 'Instances/Random_instances/experiments '+spacesize+' 03'
#     means.append(aggregated[experiment_folder]['total_times'][0])
#     stds.append(aggregated[experiment_folder]['total_times'][1])
# means = np.array(means)
# stds = np.array(stds)
    
# plt.plot(spacesizes, means, label='Mean', marker='o')
# plt.fill_between(spacesizes, means - stds, means + stds, color='lightblue', alpha=0.5, label='Std Dev')

# plt.xticks(range(7), labels)
# plt.xlabel('Search space size and 2 pipe lengths')
# plt.ylabel('Seconds required to solve')
# plt.title('Time required given different search space sizes and pipe lengths \n Given a 30% obstacle ratio.')
# plt.legend()
# plt.show()
# fig.savefig('time_vs_search_space_and_pipe_length_linechart.pdf', bbox_inches='tight')




#####################################################################################################################
# Algorithm Performance results
#####################################################################################################################

y = [50]*50+ [100]*50+ [150]*50+ [200]*50+ [250]*50+ [300]*50
x = []

fig = plt.figure(figsize=(6, 4))
experiment_pipes = [50, 100, 150, 200, 250, 300]
positions = range(1, len(experiment_pipes) + 1)  # Positions for box plots
for i, experiment_pipe in enumerate(experiment_pipes):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 ' + str(experiment_pipe) + ' 03'
    x.append(results[experiment_folder]['lengths'])
    plt.boxplot(results[experiment_folder]['lengths'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [f'{pipe} \n ' for pipe in experiment_pipes])
plt.xlabel('Pipe length setting in problem instance')
plt.ylabel('Pipe length obtained by algorithm')
plt.title('Pipe lengths given pipe lengths \n Search space $100^3$ with a 30% obstacle ratio.')

fig.savefig('Pipe_Length_vs_Pipe_Length_boxplot.pdf', bbox_inches='tight')

x = np.array(x).flatten()
y = np.array(y).flatten()
print(pearsonr(y,x))






fig = plt.figure(figsize=(6, 4))
fillratios = ['01','02','03','04','05']
positions = range(1, len(fillratios) + 1)  # Positions for box plots

for i, fill_ratio in enumerate(fillratios):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 200150 ' + fill_ratio
    plt.boxplot(results[experiment_folder]['lengths'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [str(int(ratio)*10)+'%\n ' for ratio in fillratios])
plt.xlabel('Obstacle ratio')
plt.ylabel('Pipe length obtained by algorithm')
plt.title('Pipe lengths given obstacle ratios \n Search space $100^3$ with pipe lengths 200 and 150.')

fig.savefig('Pipe_Length_vs_obstacle_ratio.pdf', bbox_inches='tight')

y = [163]*50 + [244]*50 + [325]*50 + [406]*50 + [488]*50 + [569]*50 + [650]*50
x = []
fig = plt.figure(figsize=(6, 4))
spacesizes = ['505050 10063','757575 15094','100100100 200125','125125125 250156','150150150 300188', '175175175 350219', '200200200 400250']
labels = ['$50^3$ \n 100 63','$75^3$ \n 150 94','$100^3$ \n 200 125','$125^3$ \n 250 156','$150^3$ \n 300 188', '$175^3$ \n 350 219', '$200^3$ \n 400 250']
positions = range(1, len(spacesizes) + 1)  # Positions for box plots

for i, spacesize in enumerate(spacesizes):
    experiment_folder = 'Instances/Random_instances/experiments '+spacesize+' 03'
    x.append(results[experiment_folder]['lengths'])
    plt.boxplot(results[experiment_folder]['lengths'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [size for size in labels])
plt.xlabel('Search space size and 2 pipe lengths')
plt.ylabel('Pipe length obtained by algorithm')
plt.title('Pipe length given search space sizes and pipe lengths \n Search Space with a 30% obstacle ratio.')
fig.savefig('Pipe_Length_vs_search_space_and_pipe_length.pdf', bbox_inches='tight')

x = np.array(x).flatten()
y = np.array(y).flatten()
print(pearsonr(y,x))


########################################################### nr of bends

fig = plt.figure(figsize=(6, 4))
experiment_pipes = [50, 100, 150, 200, 250, 300]
positions = range(1, len(experiment_pipes) + 1)  # Positions for box plots

for i, experiment_pipe in enumerate(experiment_pipes):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 ' + str(experiment_pipe) + ' 03'
    plt.boxplot(results[experiment_folder]['nr_bends'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [f'{pipe} \n ' for pipe in experiment_pipes])
plt.xlabel('Pipe length setting in problem instance')
plt.ylabel('Number of elbows in pipe obtained by algorithm')
plt.title('Number of elbows given pipe lengths \n Search space $100^3$ with a 30% obstacle ratio.')

fig.savefig('bends_vs_Pipe_Length_boxplot.pdf', bbox_inches='tight')


fig = plt.figure(figsize=(6, 4))
fillratios = ['01','02','03','04','05']
positions = range(1, len(fillratios) + 1)  # Positions for box plots

for i, fill_ratio in enumerate(fillratios):
    experiment_folder = 'Instances/Random_instances/experiments 100100100 200150 ' + fill_ratio
    plt.boxplot(results[experiment_folder]['nr_bends'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [str(int(ratio)*10)+'%\n ' for ratio in fillratios])
plt.xlabel('Obstacle ratio')
plt.ylabel('Number of elbows in pipe obtained by algorithm')
plt.title('Number of elbows given obstacle ratios \n Search space $100^3$ with pipe lengths 200 and 150.')

fig.savefig('bends_vs_obstacle_ratio.pdf', bbox_inches='tight')


fig = plt.figure(figsize=(6, 4))
spacesizes = ['505050 10063','757575 15094','100100100 200125','125125125 250156','150150150 300188', '175175175 350219', '200200200 400250']
labels = ['$50^3$ \n 100 63','$75^3$ \n 150 94','$100^3$ \n 200 125','$125^3$ \n 250 156','$150^3$ \n 300 188', '$175^3$ \n 350 219', '$200^3$ \n 400 250']
positions = range(1, len(spacesizes) + 1)  # Positions for box plots

for i, spacesize in enumerate(spacesizes):
    experiment_folder = 'Instances/Random_instances/experiments '+spacesize+' 03'
    plt.boxplot(results[experiment_folder]['nr_bends'], positions=[positions[i]], widths=0.6)

plt.xticks(positions, [size for size in labels])
plt.xlabel('Search space size and 2 pipe lengths')
plt.ylabel('Number of elbows in pipe obtained by algorithm')
plt.title('Number of elbows given search space and pipe lengths \n Search Space with a 30% obstacle ratio.')
fig.savefig('bends_vs_search_space_and_pipe_length.pdf', bbox_inches='tight')


#anova tests
experiment1 = results['Instances/Random_instances/experiments 100100100 50 03']['nr_bends']
experiment2 = results['Instances/Random_instances/experiments 100100100 100 03']['nr_bends']
experiment3 = results['Instances/Random_instances/experiments 100100100 150 03']['nr_bends']
experiment4 = results['Instances/Random_instances/experiments 100100100 200 03']['nr_bends']
experiment5 = results['Instances/Random_instances/experiments 100100100 250 03']['nr_bends']
experiment6 = results['Instances/Random_instances/experiments 100100100 300 03']['nr_bends']

JT_stat, z, p = jonckheere_terpstra_test([experiment2, experiment3, experiment4, experiment5, experiment6])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")
# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in elbow measurements across different pipe lengths.")
else:
    print("There is no significant difference in elbow measurements across different pipe lengths.")


f_statistic, p_value = f_oneway(experiment2, experiment3, experiment4, experiment5, experiment6)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

experiment1 = results['Instances/Random_instances/experiments 100100100 200150 01']['nr_bends']
experiment2 = results['Instances/Random_instances/experiments 100100100 200150 02']['nr_bends']
experiment3 = results['Instances/Random_instances/experiments 100100100 200150 03']['nr_bends']
experiment4 = results['Instances/Random_instances/experiments 100100100 200150 04']['nr_bends']
experiment5 = results['Instances/Random_instances/experiments 100100100 200150 05']['nr_bends']

f_statistic, p_value = f_oneway(experiment1, experiment2, experiment3, experiment4, experiment5)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in elbows measurements across different obstacle ratios.")
else:
    print("There is no significant difference in elbows measurements across different obstacle ratios.")
    
JT_stat, z, p = jonckheere_terpstra_test([experiment1, experiment2, experiment3, experiment4, experiment5])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")



experiment1 = results['Instances/Random_instances/experiments 505050 10063 03']['nr_bends']
experiment2 = results['Instances/Random_instances/experiments 757575 15094 03']['nr_bends']
experiment3 = results['Instances/Random_instances/experiments 100100100 200125 03']['nr_bends']
experiment4 = results['Instances/Random_instances/experiments 125125125 250156 03']['nr_bends']
experiment5 = results['Instances/Random_instances/experiments 150150150 300188 03']['nr_bends']
experiment6 = results['Instances/Random_instances/experiments 175175175 350219 03']['nr_bends']
experiment7 = results['Instances/Random_instances/experiments 200200200 400250 03']['nr_bends']

f_statistic, p_value = f_oneway(experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
# Interpret the result
if p_value < 0.05:
    print("There is a significant difference in elbows measurements across different search spaces.")
else:
    print("There is no significant difference in elbows measurements across different search spaces.")
   
    
JT_stat, z, p = jonckheere_terpstra_test([experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7])
print(f"JT statistic: {JT_stat:.2f}")
print(f"Z-score: {z:.2f}")
print(f"p-value: {p:.4f}")






