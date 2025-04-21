import networkx as nx
# The following code is prepared to use a networkx DiGraph,
# and evaluated the indentifiable paths given the graph and the hidden variables

def has_direct_link(G, u, v):
    # Check if there is an edge from u to v or from v to u
    return G.has_edge(u, v) or G.has_edge(v, u)

def same_confounder(G, confounder, hidden_vars):
    # returns:
    # a list with all the connected hidden confounders to the confounder
    #If there is an direct path that goes only through hidden confounders, we consider that they are the same confounder
    subgraph = G.subgraph(hidden_vars)
    same = set()
    for hidden_var in hidden_vars:
        if nx.has_path(subgraph, confounder, hidden_var):
            same.add(hidden_var)
        if nx.has_path(subgraph, hidden_var, confounder):
            same.add(hidden_var)

    return same

def find_confounded_paths(G, hidden_vars, frontdoor=False):
    """
    Find all paths between the variables in G that are confounded by hidden confounders.
    :param G: a networkx DiGraph
    :param hidden_vars: a list of hidden variables
    :param frontdoor: boolean, if True, consider the frontdoor criterion
    :return: a set of tuples with the paths that are confounded by hidden confounders
        non_confounded_set: a set of tuples with the paths that are not confounded by hidden confounders
        confounded_dict: a dict with the paths that are confounded
        frontdoor_set: a set of tuples with the paths that are identifiable following condition ii) of Prop 6.1
    """
    non_confounded_set = set()
    confounded_dict = {}  # dict needed because names of cofounders are important
    frontdoor_set = set() if frontdoor else None
    for var in G.nodes:
        if var in hidden_vars:
            continue

        # if none of the hidden confounders is parent of var, all the paths between var and its descendants are identifiable
        hidden_parents = []
        hidden_ancestors = []
        for hidden_var in hidden_vars:
            if hidden_var in G.predecessors(var):  # in nx, predecessors = parents
                hidden_parents.append(hidden_var)
            if hidden_var in nx.ancestors(G, var):
                hidden_ancestors.append(hidden_var)
        if not hidden_parents:  # if there are no hidden parents, hidden vars are not confounding the path
            if frontdoor:
                if hidden_ancestors:
                    for child in G.successors(var):
                        hidden_confounders = []
                        for hidden_ancestor in hidden_ancestors:
                            if child in G.successors(hidden_ancestor):
                                hidden_confounders.append(hidden_ancestor)
                        if hidden_confounders:
                            path = nx.shortest_path(G, var, child)
                            frontdoor_set.add(tuple(path))
                        else:
                            path = nx.shortest_path(G, var, child)
                            non_confounded_set.add(tuple(path))
                else:
                    for child in G.successors(var):
                        path = nx.shortest_path(G, var, child)
                        non_confounded_set.add(tuple(path))
            else:
                for child in G.successors(var):
                    path = nx.shortest_path(G, var, child)
                    non_confounded_set.add(tuple(path))
        else:
            # check if every path is confounded
            for child in G.successors(var):
                hidden_confounders = []
                for hidden_parent in hidden_parents:
                    if child in G.successors(hidden_parent):
                        hidden_confounders.append(hidden_parent)
                if hidden_confounders:
                    path = nx.shortest_path(G, var, child)
                    confounded_dict[tuple(path)] = hidden_confounders
                else:
                    path = nx.shortest_path(G, var, child)
                    non_confounded_set.add(tuple(path))

    return non_confounded_set, confounded_dict, frontdoor_set

def find_identifiable_interventional(G, confounded_dict, hidden_vars):
    # here we consider that we could have parents of the intervened variables and still works
    identifiable_set = set()

    for path, confounders in confounded_dict.items():

        identified_confounders = set()

        for confounder in confounders:

            parents_of_intervention = set(G.predecessors(path[0]))

            # consider that, if two confounders are interconnected, they are the same confounder
            same_set = same_confounder(G, confounder, hidden_vars)

            # if path = (parent, child)
            # we need to verify if there are at least two successors (proxies) of the hidden confounder such that
            # one proxy is d-separated from the parent given the hidden confounder
            # the other proxy is d-separated from the child given the hidden confounder
            # if we find such proxies, the path is identifiable

            # POSSIBLE ACTIVE PROXIES [condition i)]
            proxy_1_list = []  # list of proxies that are d-separated from the parent given the hidden confounder
            # NULL PROXIES [condition ii)]
            proxy_2_list = []  # list of proxies that are d-separated from the child given the hidden confounder and the treatment

            for potential_proxy in G.successors(confounder):
                if potential_proxy in path:
                    continue
                if potential_proxy in hidden_vars:
                    continue
                p = parents_of_intervention.copy()
                if potential_proxy in p:
                    p.remove(potential_proxy)
                # check if potential_proxy is d-separated from the parent given the hidden confounder
                if nx.d_separated(G, {path[0]}, {potential_proxy}, same_set.union(p)):
                    proxy_1_list.append(potential_proxy)
                elif nx.d_separated(G, {path[0], path[1]}, {potential_proxy}, same_set.union(p)):
                        proxy_1_list.append(potential_proxy)

                if nx.d_separated(G, {path[1]}, {potential_proxy},
                                  same_set.union({path[0]}).union(p)):
                    proxy_2_list.append(potential_proxy)

                # with both lists, check if there is a pair of sets that are d-separated between them given the confounder [condition i)]
                for proxy_1 in proxy_1_list:
                    for proxy_2 in proxy_2_list:
                        if proxy_1 == proxy_2:
                            continue
                        if proxy_1 in p:
                            p.remove(proxy_1)
                        if proxy_2 in p:
                            p.remove(proxy_2)
                        if nx.d_separated(G, {proxy_1}, {proxy_2}, same_set.union(p)):
                            identified_confounders.add(confounder)
                            break

        if len(identified_confounders) == len(confounders):
            identifiable_set.add(path)

    return identifiable_set



def check_identifiable_query(G, path, confounded_dict, hidden_vars, condition=None):
    '''
    Check if the query p(y | do(t), x) is identifiable given the path and the confounders
    :param G: a networkx DiGraph
    :param path: a tuple with the path (t, y)
    :param confounded_dict: a dict with the paths that are confounded
    :param hidden_vars: a list of hidden variables
    :param condition: a set of variables that are conditioned on
    :return: True if the query is identifiable, False otherwise
    '''


    t = path[0]
    y = path[1]
    x = set(condition) if condition else set()

    confounders = confounded_dict[path]
    identified_confounders = set()
    for confounder in confounders:

        # consider that, if two confounders are interconnected, they are the same confounder
        same_set = same_confounder(G, confounder, hidden_vars)

        # if path = (parent, child)
        # we need to verify if there are at least two successors (proxies) of the hidden confounder such that
        # one proxy is d-separated from the parent given the hidden confounder
        # the other proxy is d-separated from the child given the hidden confounder
        # if we find such proxies, the path is identifiable

        # Here the active has to be d-separated from the outcome. Prop A.2)
        proxy_1_list = []  # list of proxies that are d-separated from the parent given the hidden confounder
        # NULL PROXIES [condition ii)]
        proxy_2_list = []  # list of proxies that are d-separated from the child given the hidden confounder and the treatment

        for potential_proxy in G.successors(confounder):
            if potential_proxy in path:
                continue
            if potential_proxy in hidden_vars:
                continue
            p = x.copy()
            if potential_proxy in p:
                p.remove(potential_proxy)
            # check if potential_proxy is d-separated from the parent given the hidden confounder
            if nx.d_separated(G, {t}, {potential_proxy}, same_set.union(p)):
                proxy_1_list.append(potential_proxy)

            if nx.d_separated(G, {y}, {potential_proxy},
                              same_set.union({path[0]}).union(p)):
                proxy_2_list.append(potential_proxy)

        # with both lists, check if there is a pair of sets that are d-separated between them given the confounder [condition i)]
        for proxy_1 in proxy_1_list:
            for proxy_2 in proxy_2_list:
                if proxy_1 == proxy_2:
                    continue
                if nx.d_separated(G, {proxy_1}, {proxy_2}, same_set.union(x)):
                    identified_confounders.add(confounder)
                    break

    if len(identified_confounders) == len(confounders):
        return True
    return False

def check_identifiable_query_on_all_descendants(G, t, confounded_dict, hidden_vars, condition=None):
    '''check if the query p(y | do(t), x) is identifiable for all descendants of t
    :param G: a networkx DiGraph
    :param t: the treatment variable
    :param confounded_dict: a dict with the paths that are confounded
    :param hidden_vars: a list of hidden variables
    :param condition: a set of variables that are conditioned on
    :return: True if the query is identifiable, False otherwise
    '''
    for y in G.successors(t):
        path = (t, y)
        if path not in confounded_dict:
            continue
        if not check_identifiable_query(G, path, confounded_dict, hidden_vars, condition):
            return False
    return True

