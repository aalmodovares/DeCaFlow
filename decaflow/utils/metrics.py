import torch
from typing import Union, Sequence
from itertools import combinations, product
import pandas as pd

def normalize(x):
    if isinstance(x, (int, float)):
        return [x]
    elif isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return [x]
    return x

def compute_ate(flow,
                index: int | Sequence[int],
                value_a: Sequence[float],
                value_b: Sequence[float],
                num_samples: int,
                ) -> torch.Tensor:
    """
    Compute the average treatment effect (ATE) of the model.
    :param flow: decaflow / CausalFlow or SCM (is a causal flow also)
    :param index: index of the variable to intervene on
    :param value: values to intervene: 2\times len(index),
     since we need to intervene on two values and make the difference
    :param num_samples: number of samples to generate
    :return: ATE
    """

    if isinstance(index, int):
        if isinstance(value_a, torch.Tensor):
            assert value_a.dim() == 0, "value_a must be a scalar tensor."
            value_a = value_a.item()
        if isinstance(value_b, torch.Tensor):
            assert value_b.dim() == 0, "value_b must be a scalar tensor."
            value_b = value_b.item()

        assert isinstance(value_a, float) and isinstance(value_b, float)

    else:
        assert len(index) == len(value_a) and len(index) == len(value_b), \
            "Length of index and value must be the same."

    # Intervene on the first value
    x_int_a = flow.sample_interventional(index=index, value=value_a, sample_shape=(num_samples, ))
    if isinstance(x_int_a, tuple):
        x_int_a = x_int_a[0]
    # Intervene on the second value
    x_int_b = flow.sample_interventional(index=index, value=value_b, sample_shape=(num_samples, ))
    if isinstance(x_int_b, tuple):
        x_int_b = x_int_b[0]

    # Compute the mean of the two interventions
    mean_a = x_int_a.mean(0)
    mean_b = x_int_b.mean(0)

    # Compute the ATE
    ate = mean_a - mean_b
    return ate

def get_ate_error(flow, scm,
    num_hidden: int,
    index_intervene: Union[int, Sequence[int]],
    value_intervene_a: Union[float, Sequence[float]],
    value_intervene_b: Union[float, Sequence[float]],
    index_eval:Union[int, Sequence[int]]=None,
    hidden_indices:Union[int, Sequence[int]]=None,
    num_samples: int = 10000,
    scm_scale: Union[float, torch.Tensor] = 1.0, scm_loc: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:

    """
    Compute the error between the ATE of the model and the ATE of the SCM.
    :param flow: decaflow or causal flow
    :param scm: SCM
    :param num_hidden: number of hidden variables
    :param index_intervene:index to intervene or treatment (t).
    :param value_intervene: value to intervene or treatment (t).
    :param index_eval: index to evaluate the intervention or outcome(y)
        If none, all variables after the intervention are evaluated after t
    :param hidden_indices: indices of the hidden variables
        If none, all variables are evaluated
    :return: error
    :rtype: torch.Tensor
    """

    index_intervene = normalize(index_intervene)
    value_intervene_a = normalize(value_intervene_a)
    value_intervene_b = normalize(value_intervene_b)
    index_eval = normalize(index_eval) if index_eval is not None else None
    hidden_indices = normalize(hidden_indices) if hidden_indices is not None else None

    if len(index_intervene) > 1:
        raise ValueError("SCM only works with one index :(")

    assert len(value_intervene_a) == len(index_intervene), \
        "value_intervene_a must match length of index_intervene"
    assert len(value_intervene_b) == len(index_intervene), \
        "value_intervene_b must match length of index_intervene"

    if isinstance(scm_scale, float):
        assert isinstance(scm_loc, float)
    else:
        assert scm_scale.ndim == scm_loc.ndim, "scm_scale must match length of scm loc"

    if hidden_indices is not None:
        assert len(hidden_indices) == num_hidden, \
            "hidden_indices must match num_hidden"

        def adjust(idx):
            return [i - sum(h < i for h in hidden_indices) for i in idx]
    else:
        def adjust(idx):
            return [i - num_hidden for i in idx]

    index_intervene_model = adjust(index_intervene)
    index_eval_model = adjust(index_eval) if index_eval is not None else None

    # interventions in the original space
    if isinstance(scm_scale, float) or (scm_scale.ndim==0 and scm_loc.ndim==0):
        value_intervene_a_scm = [value_intervene_a[i] * scm_scale + scm_loc for i in range(len(value_intervene_a))]
        value_intervene_b_scm = [value_intervene_b[i] * scm_scale + scm_loc for i in range(len(value_intervene_b))]
    else:
        value_intervene_a_scm = [value_intervene_a_i * scm_scale[i] + scm_loc[i] for i, value_intervene_a_i in zip(index_intervene, value_intervene_a)]
        value_intervene_b_scm = [value_intervene_b_i * scm_scale[i] + scm_loc[i] for i, value_intervene_b_i in zip(index_intervene, value_intervene_b)]

    # ate_model = compute_ate(flow, index_intervene_model, value_intervene_a, value_intervene_b, num_samples)
    ate_model = compute_ate(flow, index_intervene_model[0], value_intervene_a[0], value_intervene_b[0], num_samples)
    # ate_scm = compute_ate(scm, index_intervene, value_intervene_a_scm, value_intervene_b_scm, num_samples)
    ate_scm = compute_ate(scm, index_intervene[0], value_intervene_a_scm[0], value_intervene_b_scm[0], num_samples)

    ate_scm = ate_scm / scm_scale

    if index_eval_model is not None:
        ate_model = ate_model[index_eval_model]
        ate_scm = ate_scm[index_eval]
    else:
        ate_model = ate_model[index_intervene_model[-1]:]
        ate_scm = ate_scm[index_intervene[-1]:]

    return torch.abs(ate_model - ate_scm).mean()

def get_counterfactual_error(flow, scm,
                         num_hidden: int,
                         factual: torch.Tensor,
                         index_intervene: Union[int, Sequence[int]],
                         value_intervene: Union[float, Sequence[float]],
                         index_eval: Union[int, Sequence[int]] = None,
                         hidden_indices: Union[int, Sequence[int]] = None,
                         scm_scale: Union[float, torch.Tensor] = 1.0,
                         scm_loc: Union[float, torch.Tensor] = 0.0,
                         ) -> torch.Tensor:
    """
    Compute the error between the counterfactual of the model and the counterfactual of the SCM.
    :param flow: decaflow or causal flow
    :param scm: SCM
    :param factual: factual sample
    :param index_intervene:index to intervene or treatment (t).
    :param value_intervene: value to intervene or treatment (t).
    :param index_eval: index to evaluate the intervention or outcome(y)
        If none, all variables after the intervention are evaluated after t
    :param hidden_indices: indices of the hidden variables
        If none, all variables are evaluated
    :param scm_scale: scale of the SCM // metrics in scaled space
    :param scm_loc: location of the SCM // metrics in scaled space
    :return: error
    :rtype: torch.Tensor
    """

    index_intervene = normalize(index_intervene)
    value_intervene = normalize(value_intervene)
    index_eval = normalize(index_eval) if index_eval is not None else None
    hidden_indices = normalize(hidden_indices) if hidden_indices is not None else None

    if len(index_intervene) > 1:
        raise ValueError("SCM only works with one index :(")

    assert len(value_intervene) == len(index_intervene), \
        "value_intervene must match length of index_intervene"

    if hidden_indices is not None:
        assert len(hidden_indices) == num_hidden, \
            "hidden_indices must match num_hidden"

        def adjust(idx):
            return [i - sum(h < i for h in hidden_indices) for i in idx]
    else:
        def adjust(idx):
            return [i - num_hidden for i in idx]

    index_intervene_model = adjust(index_intervene)
    index_eval_model = adjust(index_eval) if index_eval is not None else None

    # interventions in the original space
    if isinstance(scm_scale, float) or (scm_scale.ndim==0 and scm_loc.ndim==0):
        value_intervene_scm = [value_intervene[i] * scm_scale + scm_loc for i in range(len(value_intervene))]
    else:
        value_intervene_scm = [value_intervene_i * scm_scale[i] + scm_loc[i] for i, value_intervene_i in zip(index_intervene, value_intervene)]

    # The factual of the models does not contain hiddens
    if hidden_indices is not None:
        mask = torch.ones(factual.shape[1], dtype=torch.bool)
        mask[hidden_indices] = False
        factual_model = factual[:, mask]
    else:
        factual_model = factual[:, num_hidden:]

    # cf_model, _ = flow.compute_counterfactual(factual=factual_model, index=index_intervene_model, value=value_intervene)
    cf_model, _ = flow.compute_counterfactual(factual=factual_model, index=index_intervene_model[0], value=value_intervene[0])
    # cf_scm = scm.compute_counterfactual(factual=factual, index=index_intervene, value=value_intervene_scm)
    factual_scm = factual*scm_scale + scm_loc
    cf_scm = scm.compute_counterfactual(factual=factual_scm, index=index_intervene[0], value=value_intervene_scm[0])

    cf_scm = (cf_scm - scm_loc)/scm_scale

    # compare only the observational part of the the counterfactual of the scm
    if hidden_indices is not None:
        mask = torch.ones(cf_scm.shape[1], dtype=torch.bool)
        mask[hidden_indices] = False
        cf_scm = cf_scm[:, mask]
    else:
        cf_scm = cf_scm[:, num_hidden:]

    if index_eval_model is not None:
        cf_model = cf_model[:, index_eval_model]
        cf_scm = cf_scm[:, index_eval_model]
    else:
        cf_model = cf_model[:, index_intervene_model[-1]:]
        cf_scm = cf_scm[:, index_intervene_model[-1]:]
    return torch.abs(cf_model - cf_scm).mean()

def compute_kernel(x, y, kernel_type="rbf", sigma=None):
    if kernel_type == "rbf":
        if sigma is None:
            sigma = torch.median(torch.pdist(x)) + torch.median(torch.pdist(y))
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def maximum_mean_discrepancy(x, y, kernel_type="rbf", sigma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples x and y.

    Args:
        x (Tensor): A PyTorch tensor of shape (n_x, d), where n_x is the number of samples in x and d is the dimension.
        y (Tensor): A PyTorch tensor of shape (n_y, d), where n_y is the number of samples in y and d is the dimension.
        kernel_type (str): The type of kernel to use. Currently, only 'rbf' (Radial Basis Function) is supported.
        sigma (float, optional): The bandwidth parameter for the RBF kernel. If None, it will be estimated using the median heuristic.

    Returns:
        float: The MMD value between x and y.
    """
    k_xx = compute_kernel(x, x, kernel_type, sigma)
    k_yy = compute_kernel(y, y, kernel_type, sigma)
    k_xy = compute_kernel(x, y, kernel_type, sigma)

    mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    return mmd

def mmd_obs(flow, scm,
            num_hidden:int,
            num_samples:int=10000,
            hidden_indices=None):
    """
    Compute the MMD between the model and the SCM.
    :param flow: decaflow or causal flow
    :param scm: SCM
    :return: MMD
    :rtype: torch.Tensor
    """
    x = scm.sample((num_samples,))
    if hidden_indices is None:
        x = x[:, num_hidden:]
    else:
        mask = torch.zeros(x.shape[1], dtype=torch.bool)
        mask[hidden_indices] = True
        x = x[:, ~mask]
    x_flow, _ = flow.sample(num_samples=num_samples)
    return maximum_mean_discrepancy(x, x_flow)

def mmd_int(flow, scm,
           num_hidden:int,
           index_intervene:Union[int, Sequence[int]],
           value_intervene:Union[float, Sequence[float]],
           num_samples:int=10000,
           hidden_indices=None,
            scm_scale: Union[float, torch.Tensor] = 1.0,
            scm_loc: Union[float, torch.Tensor] = 0.0,
              ):
    """
    Compute the MMD between the model and the SCM.
    :param flow: decaflow or causal flow
    :param scm: SCM
    :return: MMD
    :rtype: torch.Tensor
    """
    if isinstance(scm_scale, float) or (scm_scale.ndim==0 and scm_loc.ndim==0):
        value_intervene_scm = value_intervene*scm_scale + scm_loc
    else:
        value_intervene_scm = value_intervene*scm_scale[index_intervene] + scm_loc[index_intervene]
    x = scm.sample_interventional(index=index_intervene, value=value_intervene_scm, sample_shape=(num_samples, ))
    if hidden_indices is None:
        x = x[:, num_hidden:]
    else:
        mask = torch.zeros(x.shape[1], dtype=torch.bool)
        mask[hidden_indices] = True
        x = x[:, ~mask]
    x_flow, _ = flow.sample_interventional(index=index_intervene, value=value_intervene, sample_shape=(num_samples, ))
    return maximum_mean_discrepancy(x, x_flow)

def get_ate_error_multiple_paths(flow, scm, num_hidden, quantiles, paths, train_data, hidden_indices=None,  scm_scale=1.0, scm_loc=0.0):
    """
    Define the paths to evaluate
    :param flow:
    :param scm:
    :param quantiles: List of quantiles to evaluate, all combinations of ATE computed
    :param paths: list of tuples with the indices of intervened and evaluated variables
    :param hidden_indices: list with hidden indices
    :param scm_scale: std of the scaler
    :param scm_loc: loc of the scaler
    :return: pd.DataFrame with the ATE errors of each path with each combination of quantiles
    """

    #combinations of quantiles
    percentile_combinations = [(a, b) for a, b in combinations(quantiles, 2)]
    # Generate column names
    all_combinations = [((treatment, outcome), (p1, p2)) for (treatment, outcome), (p1, p2) in product(paths, percentile_combinations)]
    columns = [
        f"ate_({treatment},{outcome})_{int(100 * p1)}-{int(100 * p2)}"
        for (treatment, outcome), (p1, p2) in all_combinations
    ]
    ate_errors = []
    for comb in all_combinations:
        (treatment, outcome), (p1, p2) = comb
        # Compute the ATE error for each combination
        value_intervene_a = train_data[:, treatment].quantile(p1)
        value_intervene_b = train_data[:, treatment].quantile(p2)
        ate_error = get_ate_error(
            flow=flow,
            scm=scm,
            num_hidden=num_hidden,
            index_intervene=treatment,
            value_intervene_a=value_intervene_a,
            value_intervene_b=value_intervene_b,
            index_eval=outcome,
            hidden_indices=hidden_indices,
            num_samples=10000,
            scm_scale=scm_scale,
            scm_loc=scm_loc
        )
        ate_errors.append(float(ate_error))

    return pd.DataFrame([ate_errors], columns = columns)


def get_counterfactual_error_multiple_paths(flow, scm,
                                            quantiles,
                                            paths,
                                            num_hidden,
                                            factual,
                                            train_data,
                                            hidden_indices=None,
                                            scm_scale=1.0,
                                            scm_loc=0.0):
    """
    Define the paths to evaluate
    :param flow:
    :param scm:
    :param quantiles: List of quantiles to evaluate, all cf computed
    :param paths: list of tuples with the indices of intervened and evaluated variables
    :param hidden_indices: list with hidden indices
    :param scm_scale: std of the scaler
    :param scm_loc: loc of the scaler
    :return: pd.DataFrame with the ATE errors of each path with each combination of quantiles
    """

    # Generate column names
    all_combinations = [((treatment, outcome), p) for (treatment, outcome), p in product(paths, quantiles)]
    columns = [
        f"cf_({treatment},{outcome})_{int(100*p)}"
        for (treatment, outcome), p in all_combinations
    ]
    cf_errors = []
    for comb in all_combinations:
        (treatment, outcome), p = comb
        # Compute the ATE error for each combination
        value_intervene = train_data[:, treatment].quantile(p)
        cf_error = get_counterfactual_error(
            flow=flow,
            scm=scm,
            num_hidden=num_hidden,
            factual=factual,
            index_intervene=treatment,
            value_intervene=value_intervene,
            index_eval=outcome,
            hidden_indices=hidden_indices,
            scm_scale=scm_scale,
            scm_loc=scm_loc
        )
        cf_errors.append(float(cf_error))

    return pd.DataFrame([cf_errors], columns=columns)


