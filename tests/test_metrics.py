from causalflows.scms import SCM, CausalEquations
import torch
import pytest
import torch
from decaflow.utils.metrics import *
class SimpleTestEquations(CausalEquations):
    def __init__(self):
        functions = [
            lambda u1: u1,  # z1
            lambda _1, u2: u2,  # z2
            lambda x1, _2, u3: x1 + u3,  # x3 = z1 + noise
            lambda _1, x2, _3, u4: x2 + u4,  # x4 = z2 + noise
            lambda x1, x2, x3, x4, u5: x3 - x4 + u5,  # t = x3 - x4 + noise
            lambda x1, x2, x3, x4, x5, u6: x3 + 2 * x5 + u6,  # y = x3 + 2t + noise
        ]

        inverses = [
            lambda x1: x1,  # z1
            lambda x1, x2: x2,  # z2
            lambda x1, x2, x3: x3 - x1,  # x3
            lambda x1, x2, x3, x4: x4 - x2,  # x4
            lambda x1, x2, x3, x4, x5: x5 - (x3 - x4),  # t
            lambda x1, x2, x3, x4, x5, x6: x6 - (x3 + 2 * x5),  # y
        ]

        super().__init__(functions, inverses)

    @property
    def adjacency(self):
        adj = torch.zeros((6, 6))
        adj[0, 2] = 1  # z1 → x1
        adj[1, 3] = 1  # z2 → x2
        adj[2, 4] = 1  # x1 → t
        adj[3, 4] = 1  # x2 → t
        adj[2, 5] = 1  # x1 → y
        adj[4, 5] = 1  # t → y
        adj += torch.eye(6)
        return adj.bool()

scm = SCM(equations=SimpleTestEquations(), base='std-gaussian')

@pytest.fixture
def simple_scm():
    return SCM(equations=SimpleTestEquations(), base='std-gaussian')

@pytest.fixture
def simple_flow(simple_scm):
    class IdentityFlow:
        def sample_interventional(self, index, value, sample_shape):
            return simple_scm.sample_interventional(index=index, value=value, sample_shape=sample_shape), None
        def compute_counterfactual(self, factual, index, value):
            return simple_scm.compute_counterfactual(factual=factual, index=index, value=value), None
        def sample(self, num_samples):
            return simple_scm.sample((num_samples,)), None
    return IdentityFlow()

def test_get_ate_error_zero(simple_flow, simple_scm):
    index = 4
    a, b = -1.0, 1.0
    error = get_ate_error(flow=simple_flow, scm=simple_scm,
                          num_hidden=0,
                          index_intervene=index,
                          value_intervene_a=a,
                          value_intervene_b=b,
                          index_eval=5,
                          num_samples=1000000)
    assert torch.isclose(error, torch.tensor(0.0), atol=1e-2),\
        f"Expected near-zero ATE error, got {error.item():.4f}"

def test_get_counterfactual_error_zero(simple_flow, simple_scm):
    factual = simple_scm.sample((100,))
    error = get_counterfactual_error(flow=simple_flow, scm=simple_scm,
                                     factual=factual,
                                     num_hidden=0,
                                     index_intervene=4,
                                     value_intervene=1.0,
                                     index_eval=5)
    assert torch.isclose(error, torch.tensor(0.0), atol=1e-2),\
        f"Expected near-zero counterfactual error, got {error.item():.4f}"


def test_mmd_obs_zero(simple_flow, simple_scm):
    mmd = mmd_obs(simple_flow, simple_scm, num_hidden=0)
    assert mmd >= 0.0
    assert mmd < 1e-2, f"Expected low MMD for observational match, got {mmd.item():.4f}"

def test_mmd_int_zero(simple_flow, simple_scm):
    mmd = mmd_int(simple_flow, simple_scm, num_hidden=0, index_intervene=5, value_intervene=1.0)
    assert mmd >= 0.0
    assert mmd < 1e-2, f"Expected low MMD for interventional match, got {mmd.item():.4f}"