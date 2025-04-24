![Logo](additional_files/logo.png)

`APRpy` is an abbreviation for Automated Pipe Routing in Python.
We have made an exact pipe routing method that, amongst others, includes multi-pipe routing and branching, while
minimizing the number of bends and the length of the pipes. In addition, we have made several instances based
on the literature that can serve as benchmarks for future research in this field.

The `APRpy` package depends only on `numpy`, `highspy`, `networkx`, and `matplotlib` packages. It can be installed
through `pip` by running the following command:

```
pip install requirements.txt
```

The backbone of `APRpy` can be found in the `src`-folder:
- `objects.py`
  - The objects.py file defines the core data structures used in the Automated Pipe Routing (APR) problem. These classes represent the various components involved in the routing process, such as pipes, connected components, and the overall APR instance. The key classes are:  
    - `Pipe`: Represents a pipe type with a unique identifier and associated costs.
    - `ConnectedComponents`: Represents a group of terminals that must be connected using a specific pipe type and certain edges.
    - `AutomatedPipeRouting`: Represents the main object containing all necessary information for an APR problem, including the graph representation of the space and the connected components.
- `parser.py`
  - This file contains functions to parse APR instances from JSON files. It includes functions to read and validate the input data, ensuring that the defined pipes, terminals, and obstacles conform to the expected format. The key steps involved are:
    - Reading JSON Data: Load the JSON file containing the APR instance definition.
    - Validating Data: Ensure that the input data is correctly formatted and that all required fields are present.
    - Creating Objects: Instantiate the `Pipe`, `ConnectedComponents`, and `AutomatedPipeRouting` objects based on the parsed data.
- `mathematical_model.py`
  - This file contains the core mathematical model for solving the Automated Pipe Routing (APR) problem. It includes functions to build and solve the optimization model, which aims to find the optimal pipe routes while minimizing the total length of pipes and the number of bends. The key steps involved are:  
    - Building the Model: Define the decision variables, objective function, and constraints for the APR problem.
    - Solving the Model: Use an optimization solver to find the optimal solution based on the defined model.
    - Extracting Results: Retrieve the optimal pipe routes, lengths, and number of bends from the solved model.
- `space_modeling.py`
  - The space_modeling.py file focuses on preprocessing the 3D space for the APR problem. It includes functions to manipulate and simplify the voxel representation of the space, as well as to create and reduce the graph representation of the space. The key steps involved are:
      - Voxel Manipulation: Remove unnecessary voxels and ensure that terminal voxels are preserved.
      - Graph Creation: Convert the 3D voxel array into a graph where nodes represent voxels and edges represent possible connections between them.
      - Graph Reduction: Simplify the graph by removing nodes and edges that are not necessary for the routing problem, while preserving the connectivity between terminals.

## Defining an Instance via JSON

### 📄 JSON Format for Defining Pipe Routing Instances

`APRpy` allows users to define benchmark or custom pipe routing problems through a structured JSON file. This makes it easy to define multiple pipes, obstacles, and routing targets in a human-readable and editable format.

#### 🔧 General Structure
```json
{
  "name": "your_instance_name",
  "size": [integer,integer,integer]
  "obstacles": [...],
  "pipes": [...]
}
```
#### Fields Breakdown
`name` (string)

A descriptive name for the instance, used for identification.

`size` (integer)

The size of the cubic 3D grid, i.e., the routing space is `[0, size-1]^3`.

`obstacles` (list of lists of integers)

Each obstacle is defined by a bounding box using 6 integers:

`[x_min, y_min, z_min, x_max, y_max, z_max]`

These define a **closed rectangular volume** in 3D space where routing is not allowed. All coordinates are zero-based and inclusive on both ends (Python-style slicing is used internally).

`pipes` (list of pipe objects)
Each pipe has the following fields:

- `id` (integer)

A unique identifier for the pipe.

- `diameter` (number, optional)

Currently unused in the core routing model, but can be used for visualization, postprocessing, or future extensions.

- `costs` (number)

The per-unit cost associated with routing this pipe.

`connected_components` (list of connected component objects)

Each connected component defines a group of terminals that must be connected via this pipe.

Connected Components

Each connected component has:

- `id` (integer)

A unique identifier for the connected component within the pipe.

- `terminals` (list of 3D coordinate lists)

A list of terminal points (e.g., start and end of a pipe, but can be more than two). All terminals must lie within the routing grid and outside any obstacles.

- `forbidden_nodes` (optional list of 3D coordinate lists)

A list of nodes (grid points) that cannot be used for this component, even if they are not part of an obstacle.

### ❗ Modeling Assumption: No Terminal Overlap
In `APRpy`, terminals must be unique across all connected components, even across different pipes. This means no two connected components (whether from the same or different pipes) may share a terminal point.

This assumption simplifies the mathematical model and avoids ambiguity during routing. If overlapping terminals are defined, an exception will be raised when initializing the model.

### ✅ Example
```json
{
  "name": "jiang_etall_case1",
  "size": 40,
  "obstacles": [
    [5, 11, 0, 23, 15, 39],
    [31, 11, 0, 39, 15, 39]
  ],
  "pipes": [
    {
      "id": 1,
      "diameter": 1,
      "costs": 1,
      "connected_components": [
        {
          "id": 1,
          "terminals": [
            [19, 0, 0],
            [19, 39, 39]
          ],
          "forbidden_nodes": []
        }
      ]
    }
  ]
}
```