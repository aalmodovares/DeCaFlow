import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def create_df(tensors_dict):
    df_list = []
    for i, (k, x) in enumerate(tensors_dict.items()):
        df = pd.DataFrame(x.numpy(), columns=[f'$x_{j}$' for j in range(x.shape[1])])
        df['name'] = k
        df_list.append(df)

    df = pd.concat(df_list)
    return df


def plot_data(tensors_dict, columns=None):
    # for k, v in tensors_dict.items():
    #     tensors_dict[k] = v * sigma + mu

    df = create_df(tensors_dict)

    df.columns = columns + ['name'] if columns is not None else df.columns

    # df = df[~df.index.duplicated()]
    g = sns.PairGrid(df, diag_sharey=False, hue='name')
    g.map_upper(sns.scatterplot, s=15, alpha=0.5)
    g.map_lower(sns.kdeplot, common_norm=False)
    g.map_diag(sns.kdeplot, lw=2, common_norm=False)
    g.add_legend()
    plt.show()
    return g

def plot_logs(logger, columns):
    fig, axs = plt.subplots(1, len(columns), figsize=(4*len(columns), 4))
    for i, col in enumerate(columns):
        sns.lineplot(x=logger.values['epoch'], y=logger.values[col], ax=axs[i])
        axs[i].set_title(col)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(col)
        axs[i].grid(True)
    plt.show()

def assign_layers(G):
    # Identify root nodes (nodes with no predecessors)
    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]

    # Dictionary to store the layer of each node
    layers = {}

    # Perform a topological sort of the graph
    topological_order = list(nx.topological_sort(G))

    # Assign layers based on the maximum distance from root nodes
    for node in topological_order:
        if node in root_nodes:
            layers[node] = 0  # Root nodes are in layer 0
        else:
            # Find the maximum layer of all parent nodes (predecessors)
            parent_layers = [layers[parent] for parent in G.predecessors(node)]
            layers[node] = max(parent_layers) + 1

    return layers


def draw_graph(G, name, hidden_vars, unconfounded_paths, interventional_id_paths,
               frontdoor_paths=None, separate_frontdoor=False, saving_name=None):

    #all sets are lists
    hidden_vars = list(hidden_vars)
    unconfounded_paths = list(unconfounded_paths)
    interventional_id_paths = list(interventional_id_paths)
    if frontdoor_paths is not None:
        frontdoor_paths = list(frontdoor_paths)

    if not separate_frontdoor:
        interventional_id_paths = interventional_id_paths + frontdoor_paths
        frontdoor_paths = []


    layers = assign_layers(G)
    special_bends = {}
    pos = {}
    layer_dict = {}


    for node, layer in layers.items():
        if layer not in layer_dict:
            layer_dict[layer] = []
        layer_dict[layer].append(node)

    max_num_nodes = max([len(nodes) for nodes in layer_dict.values()])
    x_min_spacing = 1.5  # Adjust the spacing between nodes if necessary
    len_max_row = max_num_nodes * x_min_spacing
    # Assign positions for nodes within each layer
    for layer, nodes in layer_dict.items():
        y_pos = -layer  # Y-axis position (layer number)
        num_nodes = len(nodes)
        if num_nodes==max_num_nodes:
            x_spacing = 1.5
        else:
            # x_spacing = 0.8*len_max_row / num_nodes
            x_spacing = 2

        # Calculate the starting x position for centering nodes
        x_start = -(num_nodes - 1) * x_spacing / 2

        # Assign positions to each node in the layer
        for i, node in enumerate(nodes):
            pos[node] = (x_start + i * x_spacing, y_pos)

    # Draw normal nodes
    normal_nodes = [node for node in G.nodes if node not in hidden_vars]
    # normal_nodes_collection = nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=node_size*factor, node_color='lightgrey')

    # Draw hidden nodes with dashed contours
    hidden_pos = {node: pos[node] for node in hidden_vars}


    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='lightgrey')
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_vars, node_color='white', edgecolors='black',)
    # Draw edges outgoing from hidden nodes with dashed lightgray style
    outgoing_edges = [(u, v) for u, v in G.edges if u in hidden_vars]
    nx.draw_networkx_edges(G, pos, edgelist=outgoing_edges,   edge_color='lightgray', style='dashed',arrows=True)
    # Draw unconfounded edges in solid black
    if unconfounded_paths not in [None, []]:
        unconfounded_standard_bend = []
        for i, (u, v) in enumerate(unconfounded_paths):
            # if not in special_bend
            if (u, v) not in special_bends:
                unconfounded_standard_bend.append((u, v))
        nx.draw_networkx_edges(G, pos, edgelist=unconfounded_standard_bend, edge_color='black', style='solid',
                               arrows=True)


    # draw paths with special bends
    for (u, v), sbend in special_bends.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='black', style='solid',
                               arrows=True)

    if interventional_id_paths  not in [None, []]:
        # Draw interventional paths that are NOT in counterfactual paths (dashed red)
        nx.draw_networkx_edges(G, pos, edgelist=interventional_id_paths, edge_color='green', style='solid', arrows=True)

    if frontdoor_paths  not in [None, []]:
        # draw in blue
        nx.draw_networkx_edges(G, pos, edgelist=frontdoor_paths, edge_color='green', style='solid',
                                 arrows=True)


    # Draw all other edges normally, excluding outgoing from hidden nodes, interventional paths, counterfactual paths, and unconfounded paths
    all_drawn_paths = unconfounded_paths + interventional_id_paths + outgoing_edges
    if frontdoor_paths is not None:
        all_drawn_paths = all_drawn_paths + frontdoor_paths
    other_edges = [
        (u, v) for u, v in G.edges
        if (u, v) not in all_drawn_paths
    ]
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='red', arrows=True)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes})

    # remove black edges
    plt.axis('off')

    plt.tight_layout()
    # save graph in pdg
    if saving_name is not None:
        plt.savefig(f'{saving_name}.pdf')
    plt.show()
