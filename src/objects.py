from typing import List, Tuple
import networkx as nx

class Pipe:
    """
    This object represents a pipe-type.
    """

    def __init__(self, id: int, costs: float):
        self.id = id
        self.costs = costs

    def __repr__(self):
        return f"Pipe {self.id}"

class ConnectedComponents:
    """
    This object represents the combination of terminals, edges, and pipe type. For example, terminals [1, 2, 5] must be
    connected through a certain pipe type only using certain edges.
    """

    def __init__(self, id: int, terminals: List[Tuple[int, int, int]], pipe: Pipe,
                 edges: List[Tuple[(Tuple[int, int, int], Tuple[int, int, int])]]):
        self.id = id
        self.terminals = terminals
        self.pipe = pipe
        self.edges = edges
        self.allowed_nodes = set([u for u, v in edges] + [v for u, v in edges])
        self.arcs = edges + [(v, u) for u, v in edges]
        self.root = terminals[0]

    def __repr__(self):
        return f"Terminals {self.terminals} with pipe {self.pipe}"


class AutomatedPipeRouting:
    """
    This is the main object containing all necessary information for an APR problem.
    """

    def __init__(self, connected_components: ConnectedComponents, graph: nx.Graph,
                 minimize_bends: bool, name: str = "APR instance"):
        """
        Initialize the APR object.
        :param connected_components: connected components object.
        :param graph: networkx object.
        :param minimize_bends: boolean indicating if we want to minimize the number of bends (or not).
        :param name: optional variable indicating the name of the APR instance.
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        self.arcs = list(self.graph.edges()) + [(v, u) for (u, v) in self.graph.edges()]
        self.connected_components, self.pipes, self.steiner_points_per_pipe = self.add_connected_components(connected_components)
        self.minimize_bends = minimize_bends
        self.name = name
        self.terminal_list = [cc.id for cc in self.connected_components]

    def add_connected_components(self, connected_components: ConnectedComponents):
        """
        Return the connected components, the union of their pipes, and the steiner points per pipe.
        """

        # Collect all pipes used to cover the connected_components.
        all_pipes = set(cc.pipe for cc in connected_components)

        # Check if there is no overlap in terminals between connected components with the same pipe object.
        pipe_dict = {p: set() for p in all_pipes}
        for cc in connected_components:
            # Check if the terminals are in the graph
            for terminal in cc.terminals:
                if terminal not in self.graph.nodes():
                    raise ValueError(f"Terminal {terminal} not in the graph.")

            if set(cc.terminals).intersection(pipe_dict[cc.pipe]):
                raise ValueError(f"Terminals of {cc.pipe} overlap.")
            else:
                pipe_dict[cc.pipe].update(cc.terminals)

        # Check if values from pipe_dict are not overlapping
        for p1 in pipe_dict.keys():
            for p2 in pipe_dict.keys():
                if p1 != p2:
                    if pipe_dict[p1].intersection(pipe_dict[p2]):
                        raise ValueError(f"Terminals of {p1} and {p2} overlap.")

        # Per pipe type, collect the steiner points
        steiner_points_per_pipe = {p: set(self.graph.nodes()) - pipe_dict[p] for p in all_pipes}

        # Check if the graph has a path between all terminals
        for cc in connected_components:
            for terminal1 in cc.terminals:
                for terminal2 in cc.terminals:
                    if terminal1 != terminal2:
                        if not nx.has_path(self.graph, terminal1, terminal2):
                            raise ValueError(f"No path between {terminal1} and {terminal2}.")

        return connected_components, all_pipes, steiner_points_per_pipe

    def __repr__(self):
        return self.name

