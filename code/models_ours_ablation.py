"""Two ablations of IADR: a hypernetwork that emits only the diffusion-block
weights or only the reaction-block weights.

    - Hyper-D: hypernetwork emits the HeatArray (Diffusion) weights only
              (~289 parameters); the Reaction block is a regular trainable
              nn.Module shared across all eps.
    - Hyper-R: hypernetwork emits the Reaction weights only (~47 parameters);
              the HeatArray (Diffusion) block is a regular trainable
              nn.Module shared across all eps.

Together they isolate which side of the diffusion-and-reaction split benefits
most from per-eps conditioning.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential

from nnpf.problems import AllenCahnProblem as Problem
from nnpf.problems.mean_curvature import MeanCurvatureEpsilonProblem
from nnpf.models import Reaction, HeatArray
from nnpf.utils import get_default_args

try:
    from torch.func import functional_call
except ImportError:
    try:
        from torch._functorch.functional_call import functional_call
    except ImportError:
        try:
            from torch.nn.utils.stateless import functional_call
        except ImportError:
            def functional_call(module, parameter_dict, args, kwargs=None):
                if kwargs is None:
                    kwargs = {}
                original_params = {}
                for name, param in parameter_dict.items():
                    keys = name.split('.')
                    obj = module
                    for k in keys[:-1]:
                        obj = getattr(obj, k)
                    original_params[name] = getattr(obj, keys[-1])
                    setattr(obj, keys[-1], torch.nn.Parameter(param, requires_grad=param.requires_grad))
                try:
                    if isinstance(args, tuple):
                        out = module(*args)
                    else:
                        out = module(args)
                finally:
                    for name, param in original_params.items():
                        keys = name.split('.')
                        obj = module
                        for k in keys[:-1]:
                            obj = getattr(obj, k)
                        setattr(obj, keys[-1], param)
                return out


def _build_hypernet(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _normalize_kernel_size(kernel_size, dim):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    else:
        kernel_size = list(kernel_size)
    if len(kernel_size) == 1:
        kernel_size = kernel_size * dim
    return kernel_size


# =====================================================================
# =====================================================================

class IADR_DiffOnly(MeanCurvatureEpsilonProblem):
    """Hypernet(ε) → HeatArray weights (289 params)...Reaction ..."""

    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 hyper_hidden_dim=256,
                 hyper_n_layers=3,
                 eps_min=0.0065, eps_max=0.0090,
                 norm_factor=2.0,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = _normalize_kernel_size(kernel_size, self.domain.dim)

        self.save_hyperparameters(
            'kernel_size', 'kernel_init', 'layers', 'activation',
            'hyper_hidden_dim', 'hyper_n_layers',
            'eps_min', 'eps_max', 'norm_factor',
            'train_u', 'train_eps', 'valid_u', 'valid_eps',
            'mmap', 'force_float32',
        )

        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.norm_factor = float(norm_factor)

        # ---- Diffusion: hypernet-driven (frozen template) ----
        self.diffusion_template = HeatArray(
            kernel_size=kernel_size, init=kernel_init,
            bounds=self.hparams.bounds, N=self.hparams.N,
        )
        for p in self.diffusion_template.parameters():
            p.requires_grad = False

        self._diff_param_info = []
        self.num_diff_params = 0
        for name, p in self.diffusion_template.named_parameters():
            self._diff_param_info.append((name, p.shape, p.numel()))
            self.num_diff_params += p.numel()

        self.reaction = Reaction(layers, activation)

        # ---- Hypernet ----
        self.hypernet = _build_hypernet(1, self.num_diff_params,
                                        int(hyper_hidden_dim), int(hyper_n_layers))

        # ---- info ----
        n_diff = self.num_diff_params
        n_react = sum(p.numel() for p in self.reaction.parameters())
        n_hyper = sum(p.numel() for p in self.hypernet.parameters())
        n_total_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[IADR_DiffOnly] hypernet→Diff params: {n_diff}")
        print(f"[IADR_DiffOnly] Reaction (shared trainable): {n_react}")
        print(f"[IADR_DiffOnly] hypernet itself: {n_hyper}")
        print(f"[IADR_DiffOnly] total trainable: {n_total_train}")

    # ----- helpers -----

    def _normalize_epsilon(self, epsilon):
        return 2.0 * (epsilon - self.eps_min) / (self.eps_max - self.eps_min + 1e-8) - 1.0

    # ----- forward -----

    def forward(self, x, epsilon=None):
        B = x.shape[0]
        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(-1)

        eps_normed = self._normalize_epsilon(epsilon)
        theta = self.hypernet(eps_normed)        # (B, num_diff_params)
        theta = theta * 0.1

        outputs = []
        for i in range(B):
            param_dict = {}
            ptr = 0
            for name, shape, numel in self._diff_param_info:
                param_dict[name] = theta[i, ptr:ptr + numel].view(shape)
                ptr += numel
            diff_out = functional_call(self.diffusion_template, param_dict, (x[i:i + 1],))
            react_out = self.reaction(diff_out)
            outputs.append(react_out)
        return torch.cat(outputs, dim=0)

    # ----- training / validation -----

    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({'loss': loss})

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return {'val_loss': loss}

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("IADR_DiffOnly",
            "Ablation: hypernet only generates HeatArray weights; Reaction is shared trainable.")
        group.add_argument('--kernel_size', type=int, nargs='+')
        group.add_argument('--kernel_init', choices=['zeros', 'random', 'solution'])
        group.add_argument('--layers', type=int, nargs='+')
        group.add_argument('--activation', type=str)
        group.add_argument('--hyper_hidden_dim', type=int, default=256)
        group.add_argument('--hyper_n_layers', type=int, default=3)
        group.add_argument('--eps_min', type=float, default=0.0065)
        group.add_argument('--eps_max', type=float, default=0.0090)
        group.add_argument('--norm_factor', type=float, default=2.0)

        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None)
        data.add_argument('--train_eps', type=str, default=None)
        data.add_argument('--valid_u', type=str, default=None)
        data.add_argument('--valid_eps', type=str, default=None)
        data.add_argument('--mmap', action='store_true')
        data.add_argument('--force_float32', action='store_true')

        group.set_defaults(**{**get_default_args(IADR_DiffOnly), **defaults})
        return parser


# =====================================================================
# =====================================================================

class IADR_ReactOnly(MeanCurvatureEpsilonProblem):
    """Hypernet(ε) → Reaction weights (47 params)...HeatArray ..."""

    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 hyper_hidden_dim=256,
                 hyper_n_layers=3,
                 eps_min=0.0065, eps_max=0.0090,
                 norm_factor=2.0,
                 **kwargs):
        super().__init__(**kwargs)

        kernel_size = _normalize_kernel_size(kernel_size, self.domain.dim)

        self.save_hyperparameters(
            'kernel_size', 'kernel_init', 'layers', 'activation',
            'hyper_hidden_dim', 'hyper_n_layers',
            'eps_min', 'eps_max', 'norm_factor',
            'train_u', 'train_eps', 'valid_u', 'valid_eps',
            'mmap', 'force_float32',
        )

        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.norm_factor = float(norm_factor)

        self.diffusion = HeatArray(
            kernel_size=kernel_size, init=kernel_init,
            bounds=self.hparams.bounds, N=self.hparams.N,
        )

        # ---- Reaction: hypernet-driven (frozen template) ----
        self.reaction_template = Reaction(layers, activation)
        for p in self.reaction_template.parameters():
            p.requires_grad = False

        self._react_param_info = []
        self.num_react_params = 0
        for name, p in self.reaction_template.named_parameters():
            self._react_param_info.append((name, p.shape, p.numel()))
            self.num_react_params += p.numel()

        # ---- Hypernet ----
        self.hypernet = _build_hypernet(1, self.num_react_params,
                                        int(hyper_hidden_dim), int(hyper_n_layers))

        # ---- info ----
        n_diff = sum(p.numel() for p in self.diffusion.parameters())
        n_react = self.num_react_params
        n_hyper = sum(p.numel() for p in self.hypernet.parameters())
        n_total_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[IADR_ReactOnly] HeatArray (shared trainable): {n_diff}")
        print(f"[IADR_ReactOnly] hypernet→React params: {n_react}")
        print(f"[IADR_ReactOnly] hypernet itself: {n_hyper}")
        print(f"[IADR_ReactOnly] total trainable: {n_total_train}")

    def _normalize_epsilon(self, epsilon):
        return 2.0 * (epsilon - self.eps_min) / (self.eps_max - self.eps_min + 1e-8) - 1.0

    def forward(self, x, epsilon=None):
        B = x.shape[0]
        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(-1)

        eps_normed = self._normalize_epsilon(epsilon)
        theta = self.hypernet(eps_normed)        # (B, num_react_params)
        theta = theta * 0.1

        diff_out = self.diffusion(x)             # (B, 1, H, W)

        outputs = []
        for i in range(B):
            param_dict = {}
            ptr = 0
            for name, shape, numel in self._react_param_info:
                param_dict[name] = theta[i, ptr:ptr + numel].view(shape)
                ptr += numel
            react_out = functional_call(
                self.reaction_template, param_dict, (diff_out[i:i + 1],)
            )
            outputs.append(react_out)
        return torch.cat(outputs, dim=0)

    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({'loss': loss})

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return {'val_loss': loss}

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("IADR_ReactOnly",
            "Ablation: hypernet only generates Reaction weights; HeatArray is shared trainable.")
        group.add_argument('--kernel_size', type=int, nargs='+')
        group.add_argument('--kernel_init', choices=['zeros', 'random', 'solution'])
        group.add_argument('--layers', type=int, nargs='+')
        group.add_argument('--activation', type=str)
        group.add_argument('--hyper_hidden_dim', type=int, default=256)
        group.add_argument('--hyper_n_layers', type=int, default=3)
        group.add_argument('--eps_min', type=float, default=0.0065)
        group.add_argument('--eps_max', type=float, default=0.0090)
        group.add_argument('--norm_factor', type=float, default=2.0)

        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None)
        data.add_argument('--train_eps', type=str, default=None)
        data.add_argument('--valid_u', type=str, default=None)
        data.add_argument('--valid_eps', type=str, default=None)
        data.add_argument('--mmap', action='store_true')
        data.add_argument('--force_float32', action='store_true')

        group.set_defaults(**{**get_default_args(IADR_ReactOnly), **defaults})
        return parser
