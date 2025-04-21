import numpy as np
from decaflow.utils.identifiability import find_confounded_paths, find_identifiable_interventional
from decaflow.utils.example_equations import NapkinEquations
from decaflow.utils.plot import draw_graph
from decaflow.utils.graphs import adjacency_to_nx


def test_draw_graph():

    causal_equations = NapkinEquations()
    var_names = causal_equations.var_names
    adjacency = causal_equations.adjacency
    adjacency -= np.eye(adjacency.shape[0])  # Remove self-loops
    G, hidden_vars = adjacency_to_nx(adjacency=adjacency, hidden_indices=[0,1], nodelist=var_names)

    non_confounded_paths, confounded_dict, frontdoor_set = find_confounded_paths(G, hidden_vars, frontdoor=True)
    interventional_id_paths = find_identifiable_interventional(G, confounded_dict, hidden_vars)
    draw_graph(
        G=G,
        name='napkin',
        hidden_vars=hidden_vars,
        unconfounded_paths=non_confounded_paths,
        interventional_id_paths=interventional_id_paths,
        frontdoor_paths=frontdoor_set,
        separate_frontdoor=False,
    )
