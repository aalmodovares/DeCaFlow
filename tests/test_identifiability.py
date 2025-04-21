import pytest
import torch
from decaflow.utils.identifiability import (find_confounded_paths,
                                            check_identifiable_query,
                                            check_identifiable_query_on_all_descendants)
import networkx as nx

def test_miao_identifiability():

    miao_graph = nx.DiGraph()

    miao_graph.add_edges_from([
        ('Z', 'W'),
        ('Z', 'T'),
        ('Z', 'N'),
        ('Z', 'Y'),
        ('W', 'Y'),
        ('N', 'T'),
        ('T', 'Y')
    ])
    non_confounded_set, confounded_dict, frontdoor_set  = find_confounded_paths(miao_graph, ['Z'], frontdoor=True)
    assert ('N', 'T') in confounded_dict.keys()
    assert ('W', 'Y') in confounded_dict.keys()
    assert('T', 'Y') in confounded_dict.keys()

    assert check_identifiable_query(miao_graph, ('T', 'Y'), confounded_dict, ['Z'])

    assert not check_identifiable_query(miao_graph, ('N', 'T'), confounded_dict, ['Z'])
    assert not check_identifiable_query(miao_graph, ('W', 'Y'), confounded_dict, ['Z'])

def test_all_descendants_query():

    two_outcomes_graph = nx.DiGraph()
    two_outcomes_graph.add_edges_from([
        ('Z', 'T'),
        ('Z', 'Y1'),
        ('T', 'Y1'),
        ('T', 'Y2')
    ])

    non_confounded_set, confounded_dict, frontdoor_set  = find_confounded_paths(two_outcomes_graph, ['Z'], frontdoor=True)
    assert ('T', 'Y1') in confounded_dict.keys()
    assert not ('T', 'Y2') in confounded_dict.keys()
    assert not check_identifiable_query(two_outcomes_graph, ('T', 'Y1'), confounded_dict, ['Z'])
    assert not check_identifiable_query_on_all_descendants(two_outcomes_graph, 'T', confounded_dict, ['Z'])
