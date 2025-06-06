![Banner](assets/decaflow_banner.svg)

Deconfounding Causal Flow
============
This repository is the user-friendly implementation of DeCaFlow, the deconfounding causal generative model
proposed in the paper [DeCaFlow: A Deconfounding Causal Generative Model](https://arxiv.org/abs/2503.15114).

> [!IMPORTANT]
> If you are interested in replicating the experiments reported in the paper
> — including hyperparameter searches, baselines, ablations, and other implementation details —
> please request access to the original research code. This version is more complex and was developed primarily for research purposes rather than ease of use.
Contact [alejandro.almodovar@upm.es](mailto:alejandro.almodovar@upm.es)
> or [ajavaloy@ed.ac.uk](mailto:ajavaloy@ed.ac.uk) to request access.


> [!warning]
> This repository is still under development and may contain bugs. We are still working in including more examples,
> test, experiments and documents to the repository.

For other issues related with the code or the paper, feel free to open an Issue in github or contact us.


## Installation
This repository has been created on April 2025.
At this point, the project need to be installed with the last version of the two main libraries: ``zuko>1.4.0`` and ``causalflows==0.1.0``.
Attention: the last version of causalflows only works with Python 3.11 at this point.

Therefore, first install the last version of zuko using the following command:
```bash
pip install git+https://github.com/probabilists/zuko
```

Then, install the last version of causalflows using the following command:
```bash
pip install git+https://github.com/adrianjav/causal-flows
```

Probably, if you are reading this after June 2025, you can directly install all the requirements directly.

```bash
pip install -r requirements.txt
```

To check everything is working, you can run the tests:
```bash
pytest tests/
```



## Usage
The DeCaFlow model is implemented in the `decaflow` module.
It is a lightning module that integrates the encoder and the decoder.
You can use it as follows:
```python
from decaflow.models import Encoder, Decoder, DeCaFlow
encoder = Encoder(flow_type='nsf', num_hidden=num_hidden, adjacency=adjacency,
                  features=num_hidden, context=n_features, hidden_features=[64, 64],
                  activation=torch.nn.ReLU)
decoder = Decoder(flow_type='nsf', num_hidden=num_hidden, adjacency=adjacency,
                  features=n_features, context=num_hidden, hidden_features=[64, 64, 64],
                  activation=torch.nn.ReLU)
decaflow = DeCaFlow(encoder=encoder, flow=decoder,
                    regularize=True, warmup=100,
                    lr=1e-3, optimizer_cls=torch.optim.Adam,
                    scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    scheduler_kwargs={'mode': 'min', 'factor': 0.95, 'patience': 50, 'verbose': True, 'cooldown':0},
                    scheduler_monitor='train_loss')
```

To train the model, employ a lightning trainer:
```python
import lightning as L
from decaflow.utils.logger import MyLogger
logger= MyLogger()
trainer_unaware  = L.Trainer(max_epochs=500, logger=logger , enable_checkpointing=False, log_every_n_steps=len(train_loader)-1)
trainer_unaware.fit(decaflow, train_loader)
```

The DeCaFlow model employs an ELBO and a dynamic regularization term that should be activated or not. The warm up regularization
term defines `beta=kl` when `epoch<warmup` and `beta=1` when `epoch>warmup`. The ELBO is defined as:

```elbo = log_prob_x - beta * kl_z```

where `log_prob_x` is the log-likelihood of the data and `kl_z` is the KL divergence between the prior and the posterior of the latent variables.

It allows to keep the encoder Empty, which is equivalent to using a standard Causal Flow.

```python
unaware_flow = DeCaFlow(encoder=None, flow=unaware_decoder, regularize=False,
                    lr=1e-3, optimizer_cls=torch.optim.Adam,
                    scheduler_cls=scheduler_cls,
                    scheduler_kwargs=scheduler_config,
                    scheduler_monitor='train_loss')
```

The class also implements methods for observational, interventional and counterfactual sampling.
```python
x_gen, z_gen = decaflow.sample((test_size,))
x_int, z_int = decaflow.sample_interventional(index-num_hidden, value, (test_size, ))
x_cf, z_cf = decaflow.compute_counterfactual(factual=factual, index=index_intervene, value=value])
```

To compute metrics, you can use the `metrics` module.

```python
from decaflow.utils.metrics import get_ate_error, get_counterfactual_error
ate_error = get_ate_error(flow=decaflow, scm=scm, num_hidden=num_hidden,
                      index_intervene=index_intervene,
                      value_intervene_a=value_a, value_intervene_b=value_b,
                      index_eval=index_eval)
cf_error = get_counterfactual_error(flow=decaflow, scm=scm, num_hidden=num_hidden,
                                factual = test_data, # z and x
                                index_intervene=index_intervene,
                                value_intervene=value_a,
                                index_eval=index_eval)
```

Finally, identifiability algorithms can be found in the `identifiability` module.

## Examples

Find usage examples of every module in the `notebooks` folder.

There, you can find the whole training process and the results for two example graphs.
- `napkin_example.ipynb`: Example with the Napkin graph with 2 confounders.
- `sachs_example.ipynb`: Example with the Sachs graph with 2 confounders.
- `ecoli_example.ipynb`: Example with the Ecoli graph with 46 variables and 3 confounders.

You can also find how to plot a graph like those plotted in the paper, where identifiable and non-
identifiable variables are highlighted in `draw_graphs.ipynb`.

Finally, find how to check identifiability in `identifiability_example.ipynb`.

Algorithm 5 in the paper:

```python
from decaflow.utils.identifiability import (find_confounded_paths,
                                            check_identifiable_query)
# G is a nx.DiGraph
non_confounded_set, confounded_dict, frontdoor_set  = find_confounded_paths(G, hidden_vars=hidden_vars, frontdoor=True)
is_identifiable =  check_identifiable_query(G, ('T', 'Y'), confounded_dict, hidden_vars)
```

Algorithm 6 in the paper:

```python
from decaflow.utils.identifiability import (find_confounded_paths,
                                            check_identifiable_query_on_all_descendants)
# G is a nx.DiGraph
non_confounded_set, confounded_dict, frontdoor_set  = find_confounded_paths(G, hidden_vars=hidden_vars, frontdoor=True)
check_identifiable_query_on_all_descendants(G, 'T', confounded_dict, hidden_vars)
```

## Cite as
```bibtex
@article{almodovar2025decaflow,
  title={DeCaFlow: A Deconfounding Causal Generative Model},
  author={Almod{\'o}var, Alejandro and Javaloy, Adri{\'a}n and Parras, Juan and Zazo, Santiago and Valera, Isabel},
  journal={arXiv preprint arXiv:2503.15114},
  year={2025}
}