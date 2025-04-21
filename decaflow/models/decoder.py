import torch
from typing import List
from causalflows.flows import CausalFlow, CausalMAF, CausalNAF, CausalNCSF, CausalNSF, CausalUNAF

class Decoder(CausalFlow):
    r"""Decoder class that instantiates a specific causal flow based on the flow type."""

    _flow_classes = {
        'maf': CausalMAF,
        'nsf': CausalNSF,
        'naf': CausalNAF,
        'ncsf': CausalNCSF,
        'unaf': CausalUNAF,
    }

    def __init__(self, flow_type: str, num_hidden: int, adjacency: torch.BoolTensor=None, hidden_indices:List[int]=None,
                 *args, **kwargs):
        assert flow_type in self._flow_classes, f"Unknown flow type: {flow_type}. Choose from {list(self._flow_classes)}"
        if num_hidden>0:
            if adjacency is not None:
                if hidden_indices is not None:
                    assert len(hidden_indices) == num_hidden, "Length of hidden indices must match num_hidden"
                    # Hidden variables could not be roots (not included in the theory, but useful)
                    remove_mask = torch.zeros(adjacency.shape[0], dtype=torch.bool)
                    remove_mask[hidden_indices] = True
                    adjacency_ux = adjacency[~remove_mask][:, ~remove_mask]

                    # remove hidden indices rows from adjacency_zx
                    mask = torch.zeros(adjacency.shape[0], dtype=torch.bool)
                    mask[hidden_indices] = True
                    adjacency_zx = adjacency[~mask][:, mask]
                else:
                    adjacency_ux = adjacency[num_hidden:, num_hidden:]
                    adjacency_zx = adjacency[num_hidden:, :num_hidden]

                adjacency = torch.cat((adjacency_ux, adjacency_zx), dim=1)

        flow_class = self._flow_classes[flow_type]
        flow = flow_class(adjacency = adjacency, *args, **kwargs)

        self.adjacency = adjacency

        # Initialize as a CausalFlow with the selected flow's transform and base
        super().__init__(flow.transform, flow.base)