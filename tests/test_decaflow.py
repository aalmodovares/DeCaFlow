import pytest
import torch
import lightning as L
from decaflow.models import DeCaFlow
from decaflow.models.encoder import Encoder
from decaflow.models.decoder import Decoder
from decaflow.utils.logger import MyLogger
import numpy as np

@pytest.fixture
def simple_adjacency():
    adj = torch.tensor([[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]], dtype=torch.bool)
    adj += torch.eye(4, dtype=torch.bool)
    return adj

@pytest.fixture
def encoder(simple_adjacency):
    return Encoder(
        flow_type='nsf',
        num_hidden=2,
        context=2,
        features=2,
        adjacency=simple_adjacency,
        hidden_features=[16],
        activation=torch.nn.ReLU
    )

@pytest.fixture
def decoder(simple_adjacency):
    return Decoder(
        flow_type='nsf',
        context=2,
        features=2,
        adjacency=simple_adjacency,
        num_hidden=2,
        hidden_features=[16],
        activation=torch.nn.ReLU
    )

@pytest.fixture
def dummy_batch():
    x = torch.randn(8, 2)
    return (x, )

def test_decaflow_train(encoder, decoder, dummy_batch):
    model = DeCaFlow(
        encoder=encoder,
        flow=decoder,
        regularize=True,
        warmup=10
    )
    from torch.utils.data import DataLoader, TensorDataset
    logger = MyLogger()
    trainer = L.Trainer(max_epochs=2, logger=logger, enable_checkpointing=False, log_every_n_steps=1)
    loader = DataLoader(TensorDataset(dummy_batch[0]), batch_size=2)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    train_loss, val_loss = logger.values['train_loss'][-1], logger.values['val_loss'][-1]
    assert train_loss is not None
    assert val_loss is not None

def test_sample_shapes(encoder, decoder):
    model = DeCaFlow(
        encoder=encoder,
        flow=decoder,
        regularize=False
    )
    x, z = model.sample(torch.Size([10]))
    assert x.shape[0] == 10
    if z is not None:
        assert z.shape[0] == 10

def test_sample_interventional(encoder, decoder):
    model = DeCaFlow(
        encoder=encoder,
        flow=decoder,
        regularize=False
    )
    x, z = model.sample_interventional(index=0, value=1.0, sample_shape=(5,))
    assert x.shape[0] == 5

def test_compute_counterfactual(encoder, decoder):
    model = DeCaFlow(
        encoder=encoder,
        flow=decoder,
        regularize=False
    )
    factual = torch.randn(4, 2)
    x_cf, z_cf = model.compute_counterfactual(
        factual=factual,
        index=1,
        value=2.0,
        num_samples=5
    )
    assert x_cf.shape == factual.shape

def test_compute_jacobian(encoder, decoder):
    model = DeCaFlow(
        encoder=encoder,
        flow=decoder,
        regularize=False
    )
    u = torch.randn(8, 2)
    jac = model.compute_jacobian(u=u)
    assert isinstance(jac, (torch.Tensor, np.ndarray))

    if isinstance(jac, np.ndarray):
        jac = torch.tensor(jac)
    jac_mean = jac.mean(dim=0)

    #check if it has only non-zeros in the groundtruth adjacency
    gt_adj = torch.tensor([[1, 0],
                           [1, 1]], dtype=torch.bool)
    assert (jac_mean[~gt_adj] == 0).all()