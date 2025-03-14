import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .objects import Pipe
from typing import List, Tuple, Dict

obstacle_colors = [(0, 0, 1, 0.1), (0, 1, 0, 0.1)]
pipe_colors = [(0, 0, 1, 1), (0, 1, 0, 1)]

# https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)


def plot_space_and_route(box: np.array, obstacles: np.array, result: Dict[Pipe, List[Tuple[int, int, int]]]):
    """
    Visualize the search space, obstacles, and the route of the APR-instance.
    """
    positions = obstacles[:,:3]
    sizes = obstacles[:,3:] - obstacles[:,:3]
    
    color_obstacles = [obstacle_colors[0]]*len(positions)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')
    
    pc_obstacles = plotCubeAt(positions, sizes, colors=color_obstacles, edgecolor="k")
    ax.add_collection3d(pc_obstacles)

    for res_i in result:
        pipe = result[res_i]
        pipe = np.array(pipe)
        pipesegments = []
        for pipe_segment in pipe:
            if sum(pipe_segment[0]) > sum(pipe_segment[1]):
                pipesegments.append([pipe_segment[1],pipe_segment[0]])
            else:
                pipesegments.append([pipe_segment[0],pipe_segment[1]])
        pipesegments = np.array(pipesegments)    
        positions_pipe = pipesegments[:,0]
        sizes_pipe = pipesegments[:,1] - pipesegments[:,0] + 1
        colors_pipe = [pipe_colors[1]]*len(positions_pipe)
        pc_pipe = plotCubeAt(positions_pipe, sizes_pipe, colors=colors_pipe, edgecolor="k")
        ax.add_collection3d(pc_pipe)
    
    ax.set_xlim([0,box.shape[0]])
    ax.set_ylim([0,box.shape[1]])
    ax.set_zlim([0,box.shape[2]])
    ax.set_box_aspect([box.shape[0],box.shape[1],box.shape[2]]) # to make the figure the size of the actual problem and keep aspect ratio.
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()

