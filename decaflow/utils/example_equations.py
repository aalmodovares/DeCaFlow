
from causalflows.scms import CausalEquations
import torch
import torch.distributions as D
import numpy as np
import networkx as nx
from torch.nn.functional import silu

class SachsEquations(CausalEquations):
    def __init__(self, equations_type: str = "additive"):
        if equations_type == 'nonadditive':
            raise NotImplementedError("Non-additive equations not implemented yet.")
        super().__init__(*self._build(equations_type))

    def _build(self, equations_type):
        # === 1. Define graph ===
        graph = nx.DiGraph([
            ('plcg', 'PIP3'), ('plcg', 'PIP2'), ('PIP3', 'PIP2'),
            ('PKC', 'PKA'), ('PKC', 'pjnk'), ('PKC', 'P38'),
            ('PKC', 'praf'), ('PKC', 'pmek'),
            ('PKA', 'p44/42'), ('PKA', 'pjnk'), ('PKA', 'praf'),
            ('PKA', 'pmek'), ('PKA', 'p44/42'), ('PKA', 'pakts473'),
            ('PKA', 'P38'), ('praf', 'pmek'), ('pmek', 'p44/42'), ('p44/42', 'pakts473')
        ])

        self.graph = graph

        # === 2. Topological order with PKC, PKA first ===
        nodes = ['PKC', 'PKA',
                           'plcg', 'PIP3', 'PIP2',
                           'pjnk', 'P38',
                           'praf', 'pmek', 'p44/42', 'pakts473']

        self.node_order = nodes
        # === 2. Generate weights ===
        weights = {}
        offset = int(equations_type == "nonadditive")

        for node in nodes:
            in_deg = graph.in_degree(node)
            if in_deg > 0:
                w1 = np.random.uniform(-1, 1, size=(16, in_deg + offset))
                w2 = np.random.uniform(-1, 1, size=(1, 16))
                parents = list(graph.predecessors(node))

                for i, parent in enumerate(parents):
                    if parent == "PKA":
                        w1[:, i] = np.random.uniform(low=1, high=3, size=(16,))
                        w2[:, i % 16] = np.random.uniform(low=-2, high=2, size=(1,))

                weights[node + "_1"] = w1
                weights[node + "_2"] = w2

        # === 3. Define helper functions ===
        def make_fn_additive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def fn(*xs_and_u):
                *xs, u = xs_and_u
                full = torch.stack(xs, dim=0)[parents, :]
                return (w2 @ silu(w1 @ full))[0, :] * 0.75 + u.reshape(-1)

            return fn

        def make_inverse_additive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def inv(*xs_and_y):
                *xs, y = xs_and_y
                x = torch.stack(xs, dim=0)[parents, :]
                h_x = (w2 @ silu(w1 @ x))[0, :] * 0.75
                return y - h_x  # shape: [batch_size]
            return inv

        def make_fn_nonadditive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def fn(*xs_and_u):
                *xs, u = xs_and_u
                full = torch.stack(xs, dim=0)[parents, :]
                full = torch.cat([full, u.unsqueeze(0)], dim=0)  # add noise dim: [in_dim + 1, batch_size]
                return (w2 @ silu(w1 @ full))[0, :] * 0.75
            return fn

        def make_inverse_nonadditive(w1_np, w2_np, parents):
            ...


        # === 4. Build structural functions ===
        functions = []
        inverses = []
        for node in nodes:
            parents = list(graph.predecessors(node))
            if len(parents) == 0:
                fn = lambda *xs_and_u: xs_and_u[-1]
                inv = lambda *xs_and_x: xs_and_x[-1]
            else:
                w1 = weights[node + "_1"]
                w2 = weights[node + "_2"]
                parents_indices = [nodes.index(parent) for parent in parents]
                fn = make_fn_nonadditive(w1, w2, parents_indices) \
                    if equations_type == "nonadditive" else make_fn_additive(w1, w2, parents_indices)
                inv = make_inverse_nonadditive(w1, w2, parents_indices) \
                    if equations_type == "nonadditive" else make_inverse_additive(w1, w2, parents_indices)
            functions.append(fn)
            inverses.append(inv)

        self._adjacency = self._build_adjacency(graph, nodes)
        return functions, inverses

    def _build_adjacency(self, graph, nodes):
        n = len(nodes)
        adj = torch.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        for child in nodes:
            i = node_to_idx[child]
            for parent in graph.predecessors(child):
                j = node_to_idx[parent]
                adj[i, j] = 1
        adj += torch.eye(n)
        return adj.bool()

    @property
    def adjacency(self):
        return self._adjacency

class EmpiricalGaussianBase(D.Distribution):
    def __init__(self, graph, node_order, data_sachs: "pd.DataFrame"):
        super().__init__(validate_args=False)

        self.node_order = node_order
        self.node_to_idx = {n: i for i, n in enumerate(node_order)}
        self.graph = graph

        # Identify root and non-root indices
        self.root_nodes = [n for n in node_order if graph.in_degree(n) == 0]
        self.root_indices = torch.tensor([self.node_to_idx[n] for n in self.root_nodes])
        self.non_root_indices = torch.tensor([i for i in range(len(node_order)) if i not in self.root_indices])

        # Get aligned root samples from data_sachs
        assert all(n in data_sachs.columns for n in self.root_nodes), "Mismatch in root names"
        root_data = data_sachs[self.root_nodes]
        self.data = torch.tensor(root_data.values, dtype=torch.float32)

        self.n_vars = len(node_order)
        self.gaussian_dim = len(self.non_root_indices)

        self.normal_dist = D.Normal(torch.zeros(self.gaussian_dim), torch.ones(self.gaussian_dim))
        self.categorical = D.Categorical(logits=torch.ones(len(self.data)))  # uniform over dataset rows

    def sample(self, sample_shape=torch.Size()):
        num_samples = sample_shape[0] if sample_shape else 1

        idx = self.categorical.sample((num_samples,))
        root_samples = self.data[idx]

        non_root_samples = self.normal_dist.sample((num_samples,))

        u = torch.zeros((num_samples, self.n_vars), dtype=torch.float32)
        u[:, self.root_indices] = root_samples
        u[:, self.non_root_indices] = non_root_samples

        return u

    def log_prob(self, value):
        # Not well-defined for empirical; placeholder
        raise NotImplementedError("log_prob not defined for empirical part.")

class NapkinEquations(CausalEquations):
    def __init__(self):
        s = torch.nn.functional.softplus
        self.var_names = ['Z1', 'Z2', 'W', 'B', 'T', 'Y']
        functions = [
            lambda u1: u1,  # Z1
            lambda _1, u2: u2,  # Z2
            lambda x1, x2, u3: 4 * x1 - 2 * x2 + u3,  # W
            lambda _1, _2, x3, u4: x3 + 0.5 * u4,  # B
            lambda _1, x2, _3, x4, u5: -x4 * 0.5 + x2 * 1.75 * u5,  # T
            lambda x1, _2, _3, _4, x5, u6: x1 * 4 + x5 ** 2 + u6,  # Y
        ]
        inverses = [
            lambda x1: x1,
            lambda x1, x2: x2,
            lambda x1, x2, x3: x3 - (4 * x1 - 2 * x2),  # W
            lambda _1, _2, x3, x4: 2 * (x4 - x3),  # B
            lambda _1, x2, _3, x4, x5: (x5 + x4 * 0.5) / (x2 * 1.75),  # T
            lambda x1, _2, _3, _4, x5, x6: (x6 - (x1 * 4 + x5 ** 2)),  # Y
        ]

        super().__init__(functions, inverses)

    @property
    def adjacency(self):
        adj = torch.zeros((6, 6))
        adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0])  # z1
        adj[1, :] = torch.tensor([0, 0, 0, 0, 0, 0])  # z2
        adj[2, :] = torch.tensor([1, 1, 0, 0, 0, 0])  # w
        adj[3, :] = torch.tensor([0, 0, 1, 0, 0, 0])  # b
        adj[4, :] = torch.tensor([0, 1, 0, 1, 0, 0])  # t
        adj[5, :] = torch.tensor([1, 0, 0, 0, 1, 0])  # y
        adj += torch.eye(6)
        return adj.bool()


class EcoliEquations(CausalEquations):
    def __init__(self, equations_type: str = "additive"):
        if equations_type == 'nonadditive':
            raise NotImplementedError("Non-additive equations not implemented yet.")
        super().__init__(*self._build(equations_type))

    def _build(self, equations_type):
        # === 1. Define graph ===
        graph = nx.DiGraph([
            ('b1191', 'fixC'), ('b1191', 'traA'), ('b1191', 'ygcE'),
            ('eutG', 'yceP'), ('eutG', 'ibpB'), ('eutG', 'yfaD'), ('eutG', 'lacY'), ('eutG', 'sucA'),
            ('fixC', 'ygbD'), ('fixC', 'yjbO'), ('fixC', 'cchB'), ('fixC', 'yceP'), ('fixC', 'traA'), ('fixC', 'ycgX'),
            ('sucA', 'traA'), ('sucA', 'yfaD'), ('sucA', 'ygcE'), ('sucA', 'dnaJ'), ('sucA', 'flgD'), ('sucA', 'gltA'),
            ('sucA', 'sucD'), ('sucA', 'yhdM'), ('sucA', 'atpG'), ('sucA', 'atpD'),
            ('yceP', 'b1583'), ('yceP', 'ibpB'), ('yceP', 'yfaD'),
            ('ygcE', 'icdA'), ('ygcE', 'asnA'), ('ygcE', 'atpD'),
            ('asnA', 'icdA'), ('asnA', 'lacZ'), ('asnA', 'lacA'), ('asnA', 'lacY'),
            ('cspG', 'lacA'), ('cspG', 'lacY'), ('cspG', 'yaeM'), ('cspG', 'cspA'), ('cspG', 'yeoC'), ('cspG', 'pspB'),
            ('cspG', 'yedE'), ('cspG', 'pspA'),
            ('atpD', 'yhel'),
            ('icdA', 'aceB'),
            ('lacA', 'b1583'), ('lacA', 'yaeM'), ('lacA', 'lacZ'), ('lacA', 'lacY'),
            ('cspA', 'hupB'), ('cspA', 'yfiA'),
            ('yedE', 'pspB'), ('yedE', 'pspA'), ('yedE', 'lpdA'), ('yedE', 'yhel'),
            ('lacY', 'lacZ'), ('lacY', 'nuoM'),
            ('yfiA', 'hupB'),
            ('pspB', 'pspA'),
            ('yhel', 'ycgX'), ('yhel', 'dnaG'), ('yhel', 'b1963'), ('yhel', 'folK'), ('yhel', 'dnaK'),
            ('lacZ', 'b1583'), ('lacZ', 'yaeM'), ('lacZ', 'mopB'),
            ('pspA', 'nmpC'),
            ('ycgX', 'dnaG'),
            ('dnaK', 'mopB'),
            ('mopB', 'ftsJ')
        ])

        self.graph = graph

        # === 2. Topological order with PKC, PKA first ===
        nodes = [
                    'b1191',
                    'eutG',
                    'cspG',   # unobserved confounder
                    'fixC',
                    'sucA',
                    'ygbD',
                    'yjbO',
                    'cchB',
                    'yceP',
                    'traA',
                    'ygcE',
                    'dnaJ',
                    'flgD',
                    'gltA',
                    'sucD',
                    'yhdM',
                    'atpG',
                    'ibpB',
                    'yfaD',
                    'asnA',
                    'atpD',
                    'icdA',
                    'lacA',
                    'cspA',
                    'yeoC',
                    'yedE',
                    'aceB',
                    'lacY',
                    'yfiA',
                    'pspB',
                    'lpdA',
                    'yhel',
                    'lacZ',
                    'nuoM',
                    'hupB',
                    'pspA',
                    'ycgX',
                    'b1963',
                    'folK',
                    'dnaK',
                    'b1583',
                    'yaeM',
                    'mopB',
                    'nmpC',
                    'dnaG',
                    'ftsJ'
                ]

        self.node_order = nodes
        # === 2. Generate weights ===
        weights = {}
        offset = int(equations_type == "nonadditive")

        for node in nodes:
            in_deg = graph.in_degree(node)
            if in_deg > 0:
                w1 = np.random.uniform(-1, 1, size=(16, in_deg + offset))
                w2 = np.random.uniform(-1, 1, size=(1, 16))
                parents = list(graph.predecessors(node))

                for i, parent in enumerate(parents):
                    if parent == "b1191" or parent=='eutG' or parent=='cspG':
                        w1[:, i] = np.random.uniform(low=1, high=3, size=(16,))
                        w2[:, i] = np.random.uniform(low=-2, high=2, size=(1,))

                weights[node + "_1"] = w1
                weights[node + "_2"] = w2

        # === 3. Define helper functions ===
        def make_fn_additive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def fn(*xs_and_u):
                *xs, u = xs_and_u
                full = torch.stack(xs, dim=0)[parents, :]
                return (w2 @ silu(w1 @ full))[0, :] * 0.75 + u.reshape(-1)

            return fn

        def make_inverse_additive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def inv(*xs_and_y):
                *xs, y = xs_and_y
                x = torch.stack(xs, dim=0)[parents, :]
                h_x = (w2 @ silu(w1 @ x))[0, :] * 0.75
                return y - h_x  # shape: [batch_size]
            return inv

        def make_fn_nonadditive(w1, w2, parents):
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)

            def fn(*xs_and_u):
                *xs, u = xs_and_u
                full = torch.stack(xs, dim=0)[parents, :]
                full = torch.cat([full, u.unsqueeze(0)], dim=0)  # add noise dim: [in_dim + 1, batch_size]
                return (w2 @ silu(w1 @ full))[0, :] * 0.75
            return fn

        def make_inverse_nonadditive(w1_np, w2_np, parents):
            ...


        # === 4. Build structural functions ===
        functions = []
        inverses = []
        for node in nodes:
            parents = list(graph.predecessors(node))
            if len(parents) == 0:
                fn = lambda *xs_and_u: xs_and_u[-1]
                inv = lambda *xs_and_x: xs_and_x[-1]
            else:
                w1 = weights[node + "_1"]
                w2 = weights[node + "_2"]
                parents_indices = [nodes.index(parent) for parent in parents]
                fn = make_fn_nonadditive(w1, w2, parents_indices) \
                    if equations_type == "nonadditive" else make_fn_additive(w1, w2, parents_indices)
                inv = make_inverse_nonadditive(w1, w2, parents_indices) \
                    if equations_type == "nonadditive" else make_inverse_additive(w1, w2, parents_indices)
            functions.append(fn)
            inverses.append(inv)

        self._adjacency = self._build_adjacency(graph, nodes)
        return functions, inverses

    def _build_adjacency(self, graph, nodes):
        n = len(nodes)
        adj = torch.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        for child in nodes:
            i = node_to_idx[child]
            for parent in graph.predecessors(child):
                j = node_to_idx[parent]
                adj[i, j] = 1
        adj += torch.eye(n)
        return adj.bool()

    @property
    def adjacency(self):
        return self._adjacency
