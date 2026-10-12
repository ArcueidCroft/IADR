import torch
import torch.nn as nn
from torch.nn import Sequential

# nnpf imports
from nnpf.problems import AllenCahnProblem as Problem
from nnpf.problems.mean_curvature import MeanCurvatureEpsilonProblem
from nnpf.models import Reaction, HeatArray
from nnpf.utils import get_default_args
from nnpf.nn import Parallel, LinearChannels, Residual

# try:
#     from torch.func import functional_call
# except ImportError:
#     from torch._functorch.functional_call import functional_call
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

# =====================================================================
# =====================================================================

class DR1(Problem):
    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        self.save_hyperparameters(
            'kernel_size', 'kernel_init',
            'layers', 'activation',
        )

        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(layers, activation)

        self.model = Sequential(create_diffusion(), create_reaction())

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("DR1", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        group.set_defaults(**{**get_default_args(DR1), **defaults})
        return parser


class DR2(Problem):
    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        self.save_hyperparameters(
            'kernel_size', 'kernel_init',
            'layers', 'activation',
            'train_u', 'train_eps',
            'valid_u', 'valid_eps',
            'mmap', 'force_float32', 'version',
        )

        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(layers, activation)

        self.model = Sequential(
            LinearChannels(1, 2, bias=False),
            Parallel(
                Sequential(create_diffusion(), create_reaction()),
                Residual(create_reaction()),
            ),
            LinearChannels(2, 1, bias=True),
            create_diffusion(),
            create_reaction(),
        )

    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({'loss': loss})

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        return {'val_loss': loss}

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("DR2", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None, help='train u npy path (E,S,N,1,H,W)')
        data.add_argument('--train_eps', type=str, default=None, help='train eps npy path')
        data.add_argument('--valid_u', type=str, default=None, help='valid u npy path (E,S,N,1,H,W)')
        data.add_argument('--valid_eps', type=str, default=None, help='valid eps npy path')
        data.add_argument('--mmap', action='store_true', help="np.load(..., mmap_mode='r')")
        data.add_argument('--force_float32', action='store_true', help="force u_seq float32")
        group.set_defaults(**{**get_default_args(DR2), **defaults})
        return parser


# =====================================================================
# =====================================================================

class IADR(MeanCurvatureEpsilonProblem):
    """
    DISCO ... Allen-Cahn / ...

    ...
        1. DR1 ... = Sequential(HeatArray, Reaction)...
        2. ... = MLP... epsilon... DR1 ... θ
        3. ... functional_call ... θ ... DR1 ...
        4. ... DISCO ... signed_sigmoid ...

    ...
        epsilon (B,1) → ... → MLP → θ (B, num_params)
        → signed_sigmoid(θ) → DR1_θ(u_t) → u_{t+1}

    ... DISCO ...
        - DISCO ... Transformer ... epsilon
        - DISCO ... ODE ...DR1 ...
        - ... U-Net (~200k) ... DR1 (HeatArray+Reaction, ...~...)

    ...
        ... self.hypernet...MLP...
        self.opnn...DR1 ...
    """

    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 hyper_hidden_dim=256,
                 hyper_n_layers=3,
                 eps_min=0.0065, eps_max=0.0090,
                 norm_factor=2.0,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

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

        # =============================================================
        # =============================================================
        self.opnn = Sequential(
            HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            ),
            Reaction(layers, activation),
        )

        for p in self.opnn.parameters():
            p.requires_grad = False

        self._param_info = []
        self.num_opnn_params = 0
        for name, p in self.opnn.named_parameters():
            self._param_info.append((name, p.shape, p.numel()))
            self.num_opnn_params += p.numel()

        print(f"[IADR] ...: {self.num_opnn_params}")
        for name, shape, numel in self._param_info:
            print(f"  {name}: {list(shape)} ({numel} params)")

        # =============================================================
        # =============================================================
        theta_norm = self._compute_init_norms()
        self.register_buffer("theta_norm", theta_norm)

        # =============================================================
        # =============================================================
        hidden = int(hyper_hidden_dim)
        n_layers = int(hyper_n_layers)

        mlp_layers = [nn.Linear(1, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            mlp_layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        mlp_layers.append(nn.Linear(hidden, self.num_opnn_params))
        self.hypernet = nn.Sequential(*mlp_layers)

        # nn.init.zeros_(self.hypernet[-1].weight)
        # nn.init.zeros_(self.hypernet[-1].bias)

        hyper_params = sum(p.numel() for p in self.hypernet.parameters())
        print(f"[IADR] ...: {hyper_params}")
        print(f"[IADR] ...: {hyper_params}")

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def _compute_init_norms(self) -> torch.Tensor:
        norms = []
        for name, p in self.opnn.named_parameters():
            layer_max = p.abs().max().item()
            if layer_max < 1e-8:
                layer_max = 0.01
            norms.append(torch.full((p.numel(),), layer_max))
        return torch.cat(norms)

    def _normalize_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        """... epsilon ... [-1, 1] ..."""
        return 2.0 * (epsilon - self.eps_min) / (self.eps_max - self.eps_min + 1e-8) - 1.0

    @staticmethod
    def signed_sigmoid(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        return factor * (2 * torch.sigmoid(2 * x / factor) - 1)

    def _vec_to_param_dict(self, vec: torch.Tensor) -> dict:
        param_dict = {}
        ptr = 0
        for name, shape, numel in self._param_info:
            param_dict[name] = vec[ptr:ptr + numel].view(shape)
            ptr += numel
        return param_dict

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
        """
        ... epsilon ... DR1 ...

        Args:
            x: (B, 1, H, W) ... u_t
            epsilon: (B,) ... (B,1) ... None...None ... hparams.epsilon...

        Returns:
            (B, 1, H, W) ... u_{t+1}
        """
        B = x.shape[0]

        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(-1)  # (B, 1)

        eps_normed = self._normalize_epsilon(epsilon)
        theta = self.hypernet(eps_normed)               # (B, num_opnn_params)

        theta = theta * 0.1
        outputs = []
        for i in range(B):
            param_dict = self._vec_to_param_dict(theta[i])
            out_i = functional_call(self.opnn, param_dict, (x[i:i + 1],))
            outputs.append(out_i)

        return torch.cat(outputs, dim=0)

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        ...: (data, epsilon, target1, target2, ...)
        
        ... rollout... epsilon ...
        ... DR1_θ ... target ...
        ... DISCO ...
        """
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

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("IADR", "DISCO-style hypernetwork for DR1")

        group.add_argument('--kernel_size', type=int, nargs='+',
                           help='Size of the HeatArray kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random', 'solution'],
                           help='Initialization of the HeatArray kernel')
        group.add_argument('--layers', type=int, nargs='+',
                           help='Sizes of the hidden layers in Reaction')
        group.add_argument('--activation', type=str,
                           help='Name of the activation function in Reaction')

        group.add_argument('--hyper_hidden_dim', type=int, default=256,
                           help='Hidden dimension of the hypernetwork MLP')
        group.add_argument('--hyper_n_layers', type=int, default=3,
                           help='Number of hidden layers in the hypernetwork MLP')

        group.add_argument('--eps_min', type=float, default=0.0065,
                           help='Min epsilon for normalization')
        group.add_argument('--eps_max', type=float, default=0.0090,
                           help='Max epsilon for normalization')

        # signed_sigmoid factor
        group.add_argument('--norm_factor', type=float, default=2.0,
                           help='Factor λ in signed_sigmoid (controls param range)')

        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None,
                           help='train u npy path (E,S,N,1,H,W)')
        data.add_argument('--train_eps', type=str, default=None,
                           help='train eps npy path')
        data.add_argument('--valid_u', type=str, default=None,
                           help='valid u npy path (E,S,N,1,H,W)')
        data.add_argument('--valid_eps', type=str, default=None,
                           help='valid eps npy path')
        data.add_argument('--mmap', action='store_true',
                           help="np.load(..., mmap_mode='r')")
        data.add_argument('--force_float32', action='store_true',
                           help='force u_seq float32')

        group.set_defaults(**{**get_default_args(IADR), **defaults})
        return parser


# =====================================================================
# =====================================================================

class IADR2(MeanCurvatureEpsilonProblem):
    """
    DISCO ... DR2 ...
    
    DR2 ...
        LinearChannels(1→2) → Parallel(
            Sequential(HeatArray, Reaction),
            Residual(Reaction)
        ) → LinearChannels(2→1) → HeatArray → Reaction

    ...epsilon → MLP → DR2 ... → DR2_θ(u_t) = u_{t+1}
    """

    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 hyper_hidden_dim=512,
                 hyper_n_layers=3,
                 eps_min=0.0065, eps_max=0.0090,
                 norm_factor=2.0,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

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

        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(layers, activation)

        self.opnn = Sequential(
            LinearChannels(1, 2, bias=False),
            Parallel(
                Sequential(create_diffusion(), create_reaction()),
                Residual(create_reaction()),
            ),
            LinearChannels(2, 1, bias=True),
            create_diffusion(),
            create_reaction(),
        )

        for p in self.opnn.parameters():
            p.requires_grad = False

        self._param_info = []
        self.num_opnn_params = 0
        for name, p in self.opnn.named_parameters():
            self._param_info.append((name, p.shape, p.numel()))
            self.num_opnn_params += p.numel()

        print(f"[IADR2] ...: {self.num_opnn_params}")
        for name, shape, numel in self._param_info:
            print(f"  {name}: {list(shape)} ({numel} params)")

        theta_norm = self._compute_init_norms()
        self.register_buffer("theta_norm", theta_norm)

        hidden = int(hyper_hidden_dim)
        n_layers = int(hyper_n_layers)

        mlp_layers = [nn.Linear(1, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            mlp_layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        mlp_layers.append(nn.Linear(hidden, self.num_opnn_params))
        self.hypernet = nn.Sequential(*mlp_layers)

        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)

        hyper_params = sum(p.numel() for p in self.hypernet.parameters())
        print(f"[IADR2] ...: {hyper_params}")

    def _compute_init_norms(self) -> torch.Tensor:
        norms = []
        for name, p in self.opnn.named_parameters():
            layer_max = p.abs().max().item()
            if layer_max < 1e-8:
                layer_max = 0.01
            norms.append(torch.full((p.numel(),), layer_max))
        return torch.cat(norms)

    def _normalize_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        return 2.0 * (epsilon - self.eps_min) / (self.eps_max - self.eps_min + 1e-8) - 1.0

    @staticmethod
    def signed_sigmoid(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        return factor * (2 * torch.sigmoid(2 * x / factor) - 1)

    def _vec_to_param_dict(self, vec: torch.Tensor) -> dict:
        param_dict = {}
        ptr = 0
        for name, shape, numel in self._param_info:
            param_dict[name] = vec[ptr:ptr + numel].view(shape)
            ptr += numel
        return param_dict

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]

        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(-1)

        eps_normed = self._normalize_epsilon(epsilon)
        theta = self.hypernet(eps_normed)

        theta_norm = self.theta_norm.unsqueeze(0)
        theta = theta * theta_norm / theta_norm.pow(2).mean().pow(0.5)
        theta = theta_norm * self.signed_sigmoid(theta / theta_norm, factor=self.norm_factor)

        outputs = []
        for i in range(B):
            param_dict = self._vec_to_param_dict(theta[i])
            out_i = functional_call(self.opnn, param_dict, (x[i:i + 1],))
            outputs.append(out_i)

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
        group = parser.add_argument_group("IADR2", "DISCO-style hypernetwork for DR2")

        group.add_argument('--kernel_size', type=int, nargs='+')
        group.add_argument('--kernel_init', choices=['zeros', 'random', 'solution'])
        group.add_argument('--layers', type=int, nargs='+')
        group.add_argument('--activation', type=str)
        group.add_argument('--hyper_hidden_dim', type=int, default=512)
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

        group.set_defaults(**{**get_default_args(IADR2), **defaults})
        return parser