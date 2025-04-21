import pytest
import torch
from torch import randn
from decaflow.models import Encoder

def test_posterior_factorization():

    #napkin matrix
    adj = torch.tensor([[0, 0, 0, 0, 0, 0], # z1)
                        [0, 0, 0, 0, 0, 0], # z2
                        [1, 1, 0, 0, 0, 0], # w
                        [0, 0, 1, 0, 0, 0], # b
                        [0, 1, 0, 1, 0, 0], # t
                        [1, 0, 0, 0, 1, 0]], dtype=torch.bool) # y
    adj += torch.eye(6, dtype=torch.bool)
    gt_adj = torch.tensor([[1, 0, 1, 0, 1, 1],
                           [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    num_hidden = 2
    #create encoder instance
    encoder = Encoder(flow_type='nsf', num_hidden=num_hidden, adjacency=adj,
                  features=num_hidden, context=4, hidden_features=[32, 32],
                  activation=torch.nn.ReLU)

    adj_encoder = encoder.adjacency
    #check if the adjacency matrix is equal to the groundtruth matrix
    assert torch.equal(adj_encoder, gt_adj), f"Adjacency matrix is not equal to groundtruth matrix. {adj_encoder} != {gt_adj}"



    #matrix where z1 is a child of w
    adj  = torch.tensor([[0, 0, 0, 0, 0, 0, 0], # w
                         [1, 0, 0, 0, 0, 0, 0], # z1
                         [0, 1, 0, 0, 0, 0, 0], # z2
                         [0, 0, 1, 0, 0, 0, 0], # n
                         [0, 0, 0, 0, 0, 0, 0], # b
                         [1, 1, 1, 0, 0, 0, 0], # t
                         [0, 0, 1, 0, 1, 1, 0]], dtype=torch.bool) # y
    adj += torch.eye(7, dtype=torch.bool)

    num_hidden = 2
    #groundtruth matrix
    gt_adj = torch.tensor([[1, 0, 1, 0, 0, 1, 0],
                           [1, 1, 0, 1, 1, 1, 1]
                           ], dtype=torch.bool)
    #create encoder instance
    encoder = Encoder(flow_type='nsf', num_hidden=num_hidden, adjacency=adj, hidden_indices = [1,2],
                  features=num_hidden, context=5, hidden_features=[32, 32],
                  activation=torch.nn.ReLU)

    adj_encoder = encoder.adjacency

    #check if the adjacency matrix is equal to the groundtruth matrix
    assert torch.equal(adj_encoder, gt_adj), f"Adjacency matrix is not equal to groundtruth matrix. {adj_encoder} != {gt_adj}"

def test_invalid_adjacencies():
    # not squared matrix
    invalid_adj = torch.tensor([[0, 0, 0, 0, 0, 0], # z1)
                                [0, 0, 0, 0, 0, 0], # z2
                                [1, 1, 0, 0, 0, 0], # w
                                [0, 0, 1, 0, 0, 0], # b
                                [0, 1, 0, 1, 0, 0]], dtype=torch.bool) # t
    with pytest.raises(AssertionError):
        Encoder(flow_type='nsf', num_hidden=2, adjacency=invalid_adj,
                features=2, context=4, hidden_features=[32, 32],
                activation=torch.nn.ReLU)

    # not ones in the diagonal
    invalid_adj = torch.tensor([[0, 0, 0, 0, 0, 0], # z1)
                                [0, 0, 0, 0, 0, 0], # z2
                                [1, 1, 0, 0, 0, 0], # w
                                [0, 0, 1, 1, 0, 0], # b
                                [0, 1, 0, 1, 1, 0]], dtype=torch.bool) # t
    with pytest.raises(AssertionError):
        Encoder(flow_type='nsf', num_hidden=2, adjacency=invalid_adj,
                features=2, context=4, hidden_features=[32, 32],
                activation=torch.nn.ReLU)

def test_jacobian_and_inverses():
    adj = torch.tensor([[0, 0, 0, 0, 0, 0, 0],  # w
                        [1, 0, 0, 0, 0, 0, 0],  # z1
                        [0, 1, 0, 0, 0, 0, 0],  # z2
                        [0, 0, 1, 0, 0, 0, 0],  # n
                        [0, 0, 0, 0, 0, 0, 0],  # b
                        [1, 1, 1, 0, 0, 0, 0],  # t
                        [0, 0, 1, 0, 1, 1, 0]], dtype=torch.bool)  # y
    adj += torch.eye(7, dtype=torch.bool)

    num_hidden = 2
    #groundtruth matrix
    gt_adj = torch.tensor([[1, 0],
                           [1, 1]
                           ], dtype=torch.bool)
    encoder = Encoder(flow_type='nsf', num_hidden=num_hidden, adjacency=adj, hidden_indices = [1,2],
                    features=num_hidden, context=5, hidden_features=[32, 32],
                    activation=torch.nn.ReLU)

    t = encoder.transform

    x, c = randn(2), randn(5)
    y = t(c)(x)

    y_inv = t(c).inv(y)

    assert y.shape == x.shape
    assert y.requires_grad
    assert torch.allclose(y_inv, x, atol=1e-3)

    J = torch.autograd.functional.jacobian(t(c), x)

    assert (J[~gt_adj] == 0).all()

    ladj = torch.linalg.slogdet(J).logabsdet

    assert torch.allclose(t(c).log_abs_det_jacobian(x, y), ladj, atol=1e-4)
    assert torch.allclose(J.diag().abs().log().sum(), ladj, atol=1e-4)