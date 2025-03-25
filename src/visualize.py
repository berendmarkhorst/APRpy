import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .objects import Pipe
from typing import List, Tuple, Dict

obstacle_colors = [(0, 0, 1, 0.1), (0, 0, 1, 0.3)]
pipe_colors = [(0, 0, 1, 1), (0, 1, 0, 1)]

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
    for p,s in zip(positions,sizes):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=colors, **kwargs)

def cylinder(towers, colors=None, resolution=20, **kwargs):
    """
    https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    """
    polys = []
    for tower in towers:
        center_x, center_y, start_z, radius, height_z = tower
        z = np.linspace(start_z, height_z, 2)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        # Create the sides of the cylinder
        for i in range(len(z) - 1):
            for j in range(len(theta) - 1):
                verts = [
                    [x_grid[i, j], y_grid[i, j], z_grid[i, j]],
                    [x_grid[i, j + 1], y_grid[i, j + 1], z_grid[i, j + 1]],
                    [x_grid[i + 1, j + 1], y_grid[i + 1, j + 1], z_grid[i + 1, j + 1]],
                    [x_grid[i + 1, j], y_grid[i + 1, j], z_grid[i + 1, j]]
                ]
                polys.append(verts)
        
        # Create the top face of the cylinder
        top_face = [[center_x, center_y, height_z]]
        for j in range(len(theta)):
            top_face.append([x_grid[-1, j], y_grid[-1, j], z_grid[-1, j]])
        polys.append(top_face)
        
        # Create the bottom face of the cylinder
        bottom_face = [[center_x, center_y, start_z]]
        for j in range(len(theta)):
            bottom_face.append([x_grid[0, j], y_grid[0, j], z_grid[0, j]])
        polys.append(bottom_face)
                
    # Create the Poly3DCollection
    poly3d = Poly3DCollection(polys, facecolors=colors, **kwargs)
    return poly3d

def plot_pipe(pipe):
    pipe = np.array(pipe)
    pipesegments = []
    for pipe_segment in pipe:
        if sum(pipe_segment[0]) > sum(pipe_segment[1]):
            pipesegments.append([pipe_segment[1],pipe_segment[0]]) # fix the ordering. 
        else:
            pipesegments.append([pipe_segment[0],pipe_segment[1]]) # save the already correct ordering.
    pipesegments = np.array(pipesegments)    
    positions_pipe = pipesegments[:,0]
    sizes_pipe = pipesegments[:,1] - pipesegments[:,0] + 1
    colors_pipe = [pipe_colors[1]]*len(positions_pipe)
    return plotCubeAt(positions_pipe, sizes_pipe, colors=colors_pipe, edgecolor="k")

def plot_space_and_route(box: np.array, obstacles: np.array = np.empty((0,6)), result: Dict[Pipe, List[Tuple[int, int, int]]] = {}, towers: np.array = np.empty((0,5)) ):
    """
    Visualize the search space, obstacles, and the route of the APR-instance.
    """
    positions = obstacles[:,:3]
    sizes = obstacles[:,3:] - obstacles[:,:3]
        
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_aspect('equal')
    
    if len(obstacles)>0:
        color_obstacles = [obstacle_colors[0]]*len(positions)
        pc_obstacles = plotCubeAt(positions, sizes, colors=color_obstacles, edgecolor="k")
        ax.add_collection3d(pc_obstacles)

    if len(towers)>0:
        color_cylinders = [obstacle_colors[0]]*len(towers)
        cyl_obstacles = cylinder(towers, color_cylinders, edgecolor="k")
        ax.add_collection3d(cyl_obstacles)

    for res_i in result:
        pipe = result[res_i]
        pc_pipe = plot_pipe(pipe)
        ax.add_collection3d(pc_pipe)
    
    ax.set_xlim([0,box.shape[0]])
    ax.set_ylim([0,box.shape[1]])
    ax.set_zlim([0,box.shape[2]])
    ax.set_box_aspect([box.shape[0],box.shape[1],box.shape[2]]) # to make the figure the size of the actual problem and keep aspect ratio.
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show()

def scatterplot_quick_check(search_space):
    #quick check to see if search space also is well defined in the boolean array:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the obstacles and towers
    ax.scatter(*np.where(search_space == 0), color='blue', s=1, alpha=0.1)
    
    plt.show()


