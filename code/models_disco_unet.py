"""
DISCO-style hypernetwork with a U-Net operator template (ablation baseline).

This file is a FAITHFUL reproduction of DISCO's operator network `f_theta`
(Morel, Han, Oyallon, ICML 2025; see `src/models/disco.py::OperatorNetwork`
in the official repo at https://github.com/RudyMorel/DISCO). Relative to the
official code, the ONLY change is the front-end of the hypernetwork: we
replace the Transformer encoder that ingests a short history with a simple
MLP that ingests the physical scalar epsilon. Everything downstream -- the
param-generation head, the signed_sigmoid normalisation, and the operator
template itself -- is identical to DISCO.

Operator network (matches src/models/disco.py::OperatorNetwork verbatim):
  - 4 down-/up-sampling levels; start = 8; in_channels = out_channels = 1
  - All conv layers: bias=False
  - All GroupNorm: affine=False, norm_groups=4
  - Depthwise 3x3 convolutions (groups = out_channels) inside every Down/Up
  - Residual bottleneck: x5 + bottleneck(x5), bottleneck = (1x1 -> GN -> GELU -> 1x1)
  - Periodic boundary -> padding_mode='circular' on every conv
  - No boundary mask (MCF data is on the flat torus)

Conditioning pipeline:
  epsilon (B, 1) --[MLP]--> theta (B, num_opnn_params)
  theta <- theta * theta_norm / sqrt(mean(theta_norm**2))    # whiten
  theta <- theta_norm * signed_sigmoid(theta / theta_norm)   # per-layer rescale
  output = functional_call(DiscoUNet, theta, x)

Use this file to answer the ablation question:
  "What if we keep the DISCO mechanism (eps -> MLP -> signed_sigmoid -> theta)
   but replace the DR1/DR2 physics-structured template with DISCO's own
   generic U-Net operator?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnpf.problems import AllenCahnProblem as Problem
from nnpf.problems.mean_curvature import MeanCurvatureEpsilonProblem
from nnpf.utils import get_default_args

try:
    from torch.func import functional_call
except ImportError:
    try:
        from torch._functorch.functional_call import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call


# =====================================================================
#  DISCO's operator network (U-Net) -- faithful reproduction
# =====================================================================

class _Down(nn.Module):
    """Down-sampling block (matches src/models/disco.py::Down)."""

    def __init__(self, c_in, c_mid, c_out, norm_groups, padding_mode, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=2, stride=2,
                               padding=0, padding_mode=padding_mode, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups, c_mid, affine=False)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(c_mid, c_out, kernel_size=3, stride=1,
                               padding=1, padding_mode=padding_mode,
                               groups=groups, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, c_out, affine=False)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        return x


class _Up(nn.Module):
    """Up-sampling block (matches src/models/disco.py::Up).

    Requires c_in // 2 == c_out (i.e. the skip has the same channel count
    as the up-sampled path), which holds for every level of the U-Net.
    """

    def __init__(self, c_in, c_out, norm_groups, padding_mode, groups):
        super().__init__()
        assert c_in // 2 == c_out, (
            f"_Up expects c_in // 2 == c_out, got c_in={c_in} c_out={c_out}"
        )
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2,
                                     stride=2, padding=0, bias=False)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, padding_mode=padding_mode,
                              groups=groups, bias=False)
        self.norm = nn.GroupNorm(norm_groups, c_out, affine=False)
        self.gelu = nn.GELU()

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x


class DiscoUNet(nn.Module):
    """Faithful reproduction of DISCO's OperatorNetwork (App. C.0.1).

    Structure (for start=8, the DISCO default):

        adapter_in: Conv(1, 8, k=3)                         # varying-channels
        conv_in   : Conv(8, 8, k=3) + GN + GELU
        downs[0]  : _Down(8  -> 16 -> 16)                   # dw 3x3
        downs[1]  : _Down(16 -> 32 -> 32)
        downs[2]  : _Down(32 -> 64 -> 64)
        downs[3]  : _Down(64 -> 128 -> 128)
        bottleneck: Conv(128, 256, k=1) + GN + GELU + Conv(256, 128, k=1)
        x5 <- x5 + bottleneck(x5)
        ups[0]    : _Up(128 -> 64)  # skip=64
        ups[1]    : _Up(64  -> 32)  # skip=32
        ups[2]    : _Up(32  -> 16)  # skip=16
        ups[3]    : _Up(16  -> 8)   # skip=8
        convs_out : Conv(8, 8, k=3) + GN + GELU
        adapter_out: Conv(8, 1, k=3)                        # varying-channels

    All convs use bias=False; all GroupNorms use affine=False; every 3x3 conv
    inside a Down/Up is depthwise (groups = out_channels). Depth = 4 and
    bottleneck multiplier = 2 are baked in (matches DISCO's paper and code).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        start: int = 8,
        norm_groups: int = 4,
        padding_mode: str = "circular",
    ):
        super().__init__()
        assert start % norm_groups == 0, (
            f"start ({start}) must be divisible by norm_groups ({norm_groups})"
        )
        self.start = start
        s = start

        self.adapter_in = nn.Conv2d(in_channels, s, kernel_size=3, stride=1,
                                    padding=1, padding_mode=padding_mode, bias=False)
        self.conv_in = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1,
                      padding_mode=padding_mode, bias=False),
            nn.GroupNorm(norm_groups, s, affine=False),
            nn.GELU(),
        )
        self.downs = nn.ModuleList([
            _Down(s * 1,  s * 2,  s * 2,  norm_groups, padding_mode, groups=s * 2),
            _Down(s * 2,  s * 4,  s * 4,  norm_groups, padding_mode, groups=s * 4),
            _Down(s * 4,  s * 8,  s * 8,  norm_groups, padding_mode, groups=s * 8),
            _Down(s * 8,  s * 16, s * 16, norm_groups, padding_mode, groups=s * 16),
        ])
        self.ups = nn.ModuleList([
            _Up(s * 16, s * 8, norm_groups, padding_mode, groups=s * 8),
            _Up(s * 8,  s * 4, norm_groups, padding_mode, groups=s * 4),
            _Up(s * 4,  s * 2, norm_groups, padding_mode, groups=s * 2),
            _Up(s * 2,  s * 1, norm_groups, padding_mode, groups=s * 1),
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(s * 16, s * 32, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, s * 32, affine=False),
            nn.GELU(),
            nn.Conv2d(s * 32, s * 16, kernel_size=1, bias=False),
        )
        self.convs_out = nn.Sequential(
            nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1,
                      padding_mode=padding_mode, bias=False),
            nn.GroupNorm(norm_groups, s, affine=False),
            nn.GELU(),
        )
        self.adapter_out = nn.Conv2d(s, out_channels, kernel_size=3, stride=1,
                                     padding=1, padding_mode=padding_mode, bias=False)

    def forward(self, x):
        x = self.adapter_in(x)
        x1 = self.conv_in(x)
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)
        x5 = x5 + self.bottleneck(x5)
        x = self.ups[0](x5, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        x = self.convs_out(x)
        x = self.adapter_out(x)
        return x


# =====================================================================
#  DISCO-UNet with epsilon-conditional hypernetwork
# =====================================================================

class UNet_DISCO(MeanCurvatureEpsilonProblem):
    """
    DISCO with its original U-Net operator, but conditioned on epsilon
    (via a small MLP hypernet) instead of a trajectory (via a Transformer).

    Purpose: E1 ablation for "does the physical inductive bias of DR1/DR2
    actually matter?". Keeps the DISCO mechanism (MLP(eps) -> signed_sigmoid
    normalised theta -> functional_call into a frozen operator template)
    byte-identical to IADR2 and swaps the template for DISCO's own
    U-Net (= DiscoUNet above).
    """

    def __init__(
        self,
        # U-Net template hyperparams -- start is the only "live" one;
        # depth and bottleneck_mult are kept as args for config compatibility
        # but are hard-coded to 4 / 2 inside DiscoUNet (matches DISCO paper).
        unet_base_channels: int = 8,
        unet_depth: int = 4,
        unet_bottleneck_mult: int = 2,
        # hypernetwork hyperparams
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 3,
        # epsilon normalisation range
        eps_min: float = 0.0065,
        eps_max: float = 0.0090,
        # signed_sigmoid factor (DISCO default = 2.0)
        norm_factor: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters(
            "unet_base_channels", "unet_depth", "unet_bottleneck_mult",
            "hyper_hidden_dim", "hyper_n_layers",
            "eps_min", "eps_max", "norm_factor",
            "train_u", "train_eps", "valid_u", "valid_eps",
            "mmap", "force_float32",
        )

        assert int(unet_depth) == 4, (
            "UNet_DISCO mirrors the official DISCO code which hard-codes 4 "
            "down-/up-sampling levels. Leave unet_depth=4."
        )
        assert int(unet_bottleneck_mult) == 2, (
            "UNet_DISCO mirrors the official DISCO code which hard-codes the "
            "bottleneck channel multiplier to 2 (16s -> 32s -> 16s). Leave "
            "unet_bottleneck_mult=2."
        )

        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.norm_factor = float(norm_factor)

        # -------------------------------------------------------------
        # Operator network: frozen; only its structure + init-norms are used.
        # -------------------------------------------------------------
        self.opnn = DiscoUNet(
            in_channels=1,
            out_channels=1,
            start=int(unet_base_channels),
            norm_groups=4,
            padding_mode="circular",
        )
        for p in self.opnn.parameters():
            p.requires_grad = False

        self._param_info = []
        self.num_opnn_params = 0
        for name, p in self.opnn.named_parameters():
            self._param_info.append((name, p.shape, p.numel()))
            self.num_opnn_params += p.numel()

        print(f"[UNet_DISCO] operator network parameters: {self.num_opnn_params}")
        for name, shape, numel in self._param_info:
            print(f"  {name}: {list(shape)} ({numel} params)")

        # -------------------------------------------------------------
        # Per-parameter init-norm buffer for signed_sigmoid rescaling.
        # Matches DISCO's init_normalizations: for every parameter tensor,
        # fill a tensor of the same shape with its |max| at init time, then
        # concatenate into a single (num_opnn_params,) vector.
        #
        # Because every learnable parameter in DiscoUNet is a Conv weight
        # (bias=False, GroupNorm affine=False), default Kaiming init always
        # yields a non-zero |max|, so no fallback is needed.
        # -------------------------------------------------------------
        theta_norm = self._compute_init_norms()
        self.register_buffer("theta_norm", theta_norm)

        # -------------------------------------------------------------
        # Hypernetwork: the ONLY trainable module.
        # -------------------------------------------------------------
        hidden = int(hyper_hidden_dim)
        n_layers = int(hyper_n_layers)

        mlp_layers = [nn.Linear(1, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            mlp_layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        mlp_layers.append(nn.Linear(hidden, self.num_opnn_params))
        self.hypernet = nn.Sequential(*mlp_layers)

        # NOTE: keep the default PyTorch Kaiming init on every Linear,
        # including the final one. Zero-init on the last layer would force
        # theta=0 at step 0, which under signed_sigmoid drives every U-Net
        # weight to zero and freezes the loss at variance(target) (~0.2
        # for phase-field u in [0,1]). Official DISCO does NOT zero-init
        # its param-generation head either (see param_gen_common / _adim /
        # _dim / _channels in src/models/disco.py::DISCO.__init__).

        hyper_params = sum(p.numel() for p in self.hypernet.parameters())
        print(f"[UNet_DISCO] hypernet parameters: {hyper_params}")
        print(f"[UNet_DISCO] total trainable parameters: {hyper_params}")

    # -----------------------------------------------------------------
    # Helpers (identical semantics to IADR2 / official DISCO)
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
    # Forward
    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
        """
        x:       (B, 1, H, W)
        epsilon: (B,) or (B, 1); if None, uses self.hparams.epsilon.
        returns: (B, 1, H, W)
        """
        B = x.shape[0]

        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(-1)

        eps_normed = self._normalize_epsilon(epsilon)
        theta = self.hypernet(eps_normed)  # (B, num_opnn_params)

        # DISCO normalisation (matches src/models/disco.py lines 500-504):
        #   theta <- theta * theta_norm / sqrt(mean(theta_norm**2))  # whiten
        #   theta <- theta_norm * signed_sigmoid(theta / theta_norm)  # rescale
        theta_norm = self.theta_norm.unsqueeze(0)  # (1, P)
        theta = theta * theta_norm / theta_norm.pow(2).mean().pow(0.5)
        theta = theta_norm * self.signed_sigmoid(theta / theta_norm, factor=self.norm_factor)

        # Per-sample functional_call (consistent with DR*_DISCO). vmap-ing
        # via official `functional_call_batched` is tracked in
        # todolist/2026nips/ablation_experiments.md E9.
        outputs = []
        for i in range(B):
            param_dict = self._vec_to_param_dict(theta[i])
            out_i = functional_call(self.opnn, param_dict, (x[i:i + 1],))
            outputs.append(out_i)

        return torch.cat(outputs, dim=0)

    # -----------------------------------------------------------------
    # nnpf training / validation hooks
    # -----------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({"loss": loss})

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data, epsilon=epsilon)
            loss += self.loss(data, target)
        loss /= len(targets)
        return {"val_loss": loss}

    # -----------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group(
            "UNet_DISCO",
            "DISCO-style hypernetwork with the official DISCO U-Net operator "
            "(only the Transformer encoder is replaced with an MLP(eps)).",
        )

        group.add_argument("--unet_base_channels", type=int, default=8,
                           help="Starting channel count `start` of the U-Net (DISCO default: 8).")
        group.add_argument("--unet_depth", type=int, default=4,
                           help="Number of down-/up-sampling levels. MUST be 4 (DISCO hard-codes).")
        group.add_argument("--unet_bottleneck_mult", type=int, default=2,
                           help="Bottleneck channel multiplier. MUST be 2 (DISCO hard-codes).")

        group.add_argument("--hyper_hidden_dim", type=int, default=128,
                           help="Hidden dim of the hypernetwork MLP.")
        group.add_argument("--hyper_n_layers", type=int, default=3,
                           help="Number of hidden layers in the hypernetwork MLP.")

        group.add_argument("--eps_min", type=float, default=0.0065,
                           help="Min epsilon for normalisation.")
        group.add_argument("--eps_max", type=float, default=0.0090,
                           help="Max epsilon for normalisation.")
        group.add_argument("--norm_factor", type=float, default=2.0,
                           help="Factor lambda in signed_sigmoid (DISCO default: 2.0).")

        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument("--train_u", type=str, default=None)
        data.add_argument("--train_eps", type=str, default=None)
        data.add_argument("--valid_u", type=str, default=None)
        data.add_argument("--valid_eps", type=str, default=None)
        data.add_argument("--mmap", action="store_true")
        data.add_argument("--force_float32", action="store_true")

        group.set_defaults(**{**get_default_args(UNet_DISCO), **defaults})
        return parser


# =====================================================================
#  Smoke test
# =====================================================================

if __name__ == "__main__":
    net = DiscoUNet(in_channels=1, out_channels=1, start=8)
    x = torch.randn(2, 1, 256, 256)
    y = net(x)
    n = sum(p.numel() for p in net.parameters())
    print(f"DiscoUNet output: {y.shape}, params: {n} ({n/1e3:.1f}k)")
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
