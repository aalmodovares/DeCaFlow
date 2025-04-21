import torch
from causalflows.scms import SCM
from cdt.data import load_dataset
import pytest
from decaflow.utils.example_equations import SachsEquations, EmpiricalGaussianBase

@pytest.mark.parametrize("equations_type", ["additive"])
def test_sachs(equations_type):
    data_sachs, _ = load_dataset("sachs")
    eq = SachsEquations(equations_type = equations_type)
    graph = eq.graph
    node_order = eq.node_order
    base_dist = EmpiricalGaussianBase(graph, node_order, data_sachs)
    u_samples = base_dist.sample((128,))
    scm =  SCM(equations=eq, base=base_dist)

    adjacency_gt = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pkc
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pka
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # plcg
                                 [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # pip3
                                 [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], # pip2
                                 [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], # jnk
                                 [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], # p38
                                 [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], # raf
                                 [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0], # mek
                                 [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0], # erk
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], # akt

    ], dtype=torch.bool)

    adjacency = eq.adjacency
    assert torch.equal(adjacency, adjacency_gt), "Adjacency matrix does not match the expected adjacency matrix"

    # test if u == transform(inverse(u))

    x = scm.transform.inv(u_samples)
    u = scm.transform(x)
    assert torch.allclose(u, u_samples, atol=1e-3), "Inverse transform failed"
