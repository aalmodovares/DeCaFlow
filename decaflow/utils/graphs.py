import networkx as nx
import torch
from typing import List, Union

def nx_to_adjacency(graph: nx.DiGraph, hidden_vars:Union[List[int],List[str]]  = None) -> [torch.Tensor, List[int]]:
    """
    Convert a NetworkX graph to an adjacency matrix, excluding hidden variables.

    Args:
        graph (networkx.Graph): The input graph.
        hidden_vars (list): List of indices or names of hidden variables.

    Returns:
        torch.Tensor: Adjacency matrix of the graph.
    """
    adjacency = nx.to_numpy_array(graph)
    adjacency = torch.tensor(adjacency, dtype=torch.bool)

    if isinstance(hidden_vars[0], int):
        return adjacency, hidden_vars
    else:
        # Convert hidden variable names to indices
        var_names = list(graph.nodes())
        hidden_indices = [var_names.index(var) for var in hidden_vars]

        return adjacency, hidden_indices


def adjacency_to_nx(adjacency: torch.Tensor, hidden_indices:List[int] = None, nodelist:List[str]=None
                    )-> [nx.DiGraph, Union[List[int], List[str]]]:
    """
    Convert an adjacency matrix to a NetworkX graph, adding hidden variables.

    Args:
        adjacency (torch.Tensor): The input adjacency matrix.
        hidden_indices (list): List of indices of hidden variables.

    Returns:
        networkx.Graph: The resulting graph.
    """
    adjacency = adjacency.numpy().transpose()
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph, nodelist=nodelist)
    if nodelist is not None:
        hidden_vars = [list(graph.nodes())[i] for i in hidden_indices]
    else:
        hidden_vars = hidden_indices
    return graph, hidden_vars



