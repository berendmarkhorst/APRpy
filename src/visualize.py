import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .objects import Pipe
from typing import List, Tuple, Dict

colors = [(0, 0, 1, 0.5), (0, 1, 0, 0.5)]


def plot_space_and_route(box: np.array, result: Dict[Pipe, List[Tuple[int, int, int]]]):
    """
    Visualize the search space, obstacles, and the route of the APR-instance.
    """
    # Step 3: Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles using ax.voxels
    voxels = (box == 0)

    # Create an empty 3D colors array (with the same shape as box) of type object
    colors_array = np.full(box.shape, None, dtype=object)

    # Count how many obstacles there are
    n_voxels = np.count_nonzero(voxels)

    # Create a 1D object array with each element being the tuple (1, 0, 0, 1)
    colors_list = np.empty(n_voxels, dtype=object)
    colors_list.fill((1, 0, 0, 1))

    # Assign the 1D colors_list to the positions where voxels is True
    colors_array[voxels] = colors_list

    ax.voxels(voxels, facecolors=colors_array, edgecolor='k')

    for idx, pipe in enumerate(result.keys()):
        points = np.array([pt for pt in result[pipe]])
        for pt in points:
            x, y, z = pt
            ax.bar3d(x, y, z, 1, 1, 1, color=colors[idx], alpha=0.5, edgecolor='k')

    # Step 6: Adjust plot settings
    ax.set_xlim([0, box.shape[0]])
    ax.set_ylim([0, box.shape[1]])
    ax.set_zlim([0, box.shape[2]])

    plt.show()
