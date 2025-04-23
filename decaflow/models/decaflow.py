import torch
from torch import distributions as D
import lightning as L
from typing import List, Tuple, Sequence, Union

from zuko.flows import Flow
from causalflows.flows import CausalFlow


class DeCaFlow(L.LightningModule):
    def __init__(self,
                 encoder: Union[Flow, None],
                 flow: CausalFlow,
                 regularize: bool,
                 warmup: int = None,
                 lr: float = 1e-3,
                 optimizer_cls: type = torch.optim.Adam,
                 optimizer_kwargs: dict = None,
                 scheduler_cls: type = None,
                 scheduler_kwargs: dict = None,
                 scheduler_monitor: str = None):
        '''
        :param encoder: Conditional Normalizing Flow encoder. Can be causal or non-causal.
          If None, the model is a Causal Normalizing Flow
        :param flow: Causal Normalizing flow decoder
        :param regularize: bool, apply KL regularization to avoid posterior collapse
        :param warmup: int, number of epochs to warm up the KL regularization
        '''
        super().__init__()
        self.encoder = encoder
        self.flow = flow # decoder
        self.automatic_optimization = False

        self.regularize = regularize
        self.warmup = warmup

        if self.regularize:
            assert self.warmup is not None, "Regularization is set to True, but warmup is not set."

        # Optimizer and scheduler configuration
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_monitor = scheduler_monitor

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.lr,
            **self.optimizer_kwargs
        )

        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": True
            }
            return [optimizer], [scheduler_config]

        return optimizer

    def sample_prior(self, sample_shape: torch.Size) -> torch.Tensor:
        prior = D.Independent(
                D.Normal(loc=torch.zeros(self.encoder.latent_dim).to(self.device), scale=1),
                reinterpreted_batch_ndims=1)
        return prior.sample(sample_shape)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x = batch[0]

        x = x.flatten(start_dim=1)

        if self.encoder is not None:
            q_z = self.encoder(x)
            z = q_z.rsample()

            p_z = D.Normal(torch.zeros_like(z), scale=1)

            kl_z = q_z.log_prob(z) - p_z.log_prob(z).sum(dim=-1)  # Monte Carlo estimate of KL divergence

            beta = 1.
            if self.regularize and self.trainer.current_epoch < self.warmup:
                beta = min(1., kl_z.mean() / z.shape[1])
        else:
            z = None

        log_prob_x = self.flow(z).log_prob(x)
        if self.encoder is not None:
            elbo = log_prob_x - beta * kl_z
            loss = -elbo.mean(dim=0)
        else:
            loss = -log_prob_x.mean(dim=0)

        if getattr(self.trainer, "logger", None) is not None:
            self.log('train_loss', loss.detach(), prog_bar=True)
            self.log('log_prob', log_prob_x.mean(dim=0).detach(), prog_bar=True)
            if self.encoder is not None:
                self.log('kl_z', kl_z.mean(dim=0).detach(), prog_bar=True)
            if self.trainer.lr_scheduler_configs:
                self.log('lr', self.lr_schedulers().get_last_lr()[-1], prog_bar=True)

        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x = x.flatten(start_dim=1)

        if self.encoder is not None:
            q_z = self.encoder(x)
            z = q_z.rsample()

            p_z = D.Normal(torch.zeros_like(z), scale=1)

            kl_z = q_z.log_prob(z) - p_z.log_prob(z).sum(dim=-1)
            beta = 1. #always 1 for validation
        else:
            z = None
            kl_z = None
            beta = 0.

        log_prob_x = self.flow(z).log_prob(x)
        if self.encoder is not None:
            elbo = log_prob_x - beta * kl_z
            loss = -elbo.mean(dim=0)
        else:
            loss = -log_prob_x.mean(dim=0)
        if getattr(self.trainer, "logger", None) is not None:
            self.log('val_loss', loss.detach(), on_step=False, on_epoch=True,  prog_bar=True)
            self.log('val_log_prob', log_prob_x.mean(dim=0).detach(), on_step=False, on_epoch=True, prog_bar=False)
            if self.encoder is not None:
                self.log('val_kl_z', kl_z.mean(dim=0).detach(), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def sample(self, sample_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model.
        :param num_samples: number of samples to generate
        :return: tuple of (samples, latent variables)
        """
        z = self.sample_prior(sample_shape) if self.encoder is not None else None
        if z is None:
            x = self.flow().sample(sample_shape)
        else:
            x = self.flow(z).sample()
        return x, z

    def sample_interventional(self,
                              index: torch.LongTensor | int | Sequence[int],
                              value: torch.Tensor | float | Sequence[float],
                              sample_shape: torch.Size = (),
                            ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Sample from the model with interventional variables.
        :param index: index of the variable to intervene on
        :param value: value to set the variable to
        :param num_samples: number of samples to generate
        :return: samples with the intervened variables
        """
        z = self.sample_prior(sample_shape) if self.encoder is not None else None
        if z is None:
            x = self.flow().sample_interventional(index=index, value=value, sample_shape=sample_shape)
        else:
            x = self.flow(z).sample_interventional(index=index, value=value, )
        return x, z

    def compute_counterfactual(self,
                               factual: torch.Tensor,
                               index: torch.LongTensor | int | Sequence[int],
                               value: torch.Tensor | float | Sequence[float],
                               num_samples: int = 10,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the counterfactual samples.
        :param factual: factual samples
        :param index: index of the variable to intervene on
        :param value: value to set the variable to
        :param num_samples: number of samples of Z to generate to take the mean
        :return: counterfactual samples and samples of posterior of Z
        """
        if self.encoder is not None:
            z  = self.encoder(factual).sample((num_samples,))
            z = z.mean(dim=0)
        else:
            z = None
        x = self.flow(z).compute_counterfactual(factual=factual, index=index, value=value)
        return x, z

    def compute_jacobian(self,
                         x: torch.Tensor = None,
                         u: torch.Tensor = None,
                         ) -> torch.Tensor:
        assert x is not None or u is not None, "Either x or u must be provided."
        num_samples = x.shape[0] if x is not None else u.shape[0]
        z = self.sample_prior((num_samples, )) if self.encoder is not None else None

        n_flow = self.flow(z)

        with torch.enable_grad():
            if x is not None:  # Compute Jacobian at x
                fn = n_flow.transform
                jac = torch.autograd.functional.jacobian(fn, x.mean(0)).cpu().numpy()
            else:  # Compute Jacobian at u
                fn = n_flow.transform.inv
                jac = torch.autograd.functional.jacobian(fn, u.mean(0)).cpu().numpy()

        return jac




