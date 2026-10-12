"""
FNO baseline for phase-field mean curvature flow with epsilon conditioning.
Drop-in replacement for DR2_Epsilon: same forward(x, epsilon) signature,
same base class (MeanCurvatureEpsilonProblem).

Two variants provided:
  1) FNO_Epsilon          ... epsilon concatenated as extra input channel (reviewer's "naive" baseline)
  2) FNO_Epsilon_FiLM     ... epsilon injected via FiLM modulation after each Fourier layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import math

from nnpf.problems import AllenCahnProblem as Problem
from nnpf.problems.mean_curvature import MeanCurvatureEpsilonProblem
from nnpf.utils import get_default_args


# =========================================================================
#  Core FNO building blocks
# =========================================================================

class SpectralConv2d(nn.Module):
    """2-D Fourier layer: learnable weight in truncated Fourier space."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of Fourier modes to keep (height)
        self.modes2 = modes2  # number of Fourier modes to keep (width)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def _complex_mul2d(a, b):
        # a: (B, Ci, M1, M2), b: (Ci, Co, M1, M2)  ... both complex
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        B = x.shape[0]
        # rfft2 -> (B, C, H, W//2+1)  complex
        x_ft = torch.fft.rfft2(x)

        w1 = torch.view_as_complex(self.weights1)  # (Ci, Co, M1, M2)
        w2 = torch.view_as_complex(self.weights2)

        out_ft = torch.zeros(B, self.out_channels, x_ft.size(2), x_ft.size(3),
                             dtype=x_ft.dtype, device=x.device)

        # low-freq corner (top-left)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        # low-freq corner (bottom-left, negative freqs in dim-1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

        return torch.fft.irfft2(out_ft, s=(x.size(2), x.size(3)))


class FourierBlock(nn.Module):
    """One Fourier layer: spectral conv + 1x1 conv (residual path) + activation."""
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.conv(x))


# =========================================================================
#  Variant 1: FNO with epsilon as extra input channel  (naive concat)
# =========================================================================

class FNO_Epsilon(MeanCurvatureEpsilonProblem):
    """
    Standard FNO where epsilon is broadcast to a constant spatial map
    and concatenated as a 2nd input channel:  input = [u, eps_map]  (B,2,H,W).
    """
    def __init__(self,
                 fno_width=32,
                 fno_modes=12,
                 fno_layers=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            'fno_width', 'fno_modes', 'fno_layers',
            'train_u', 'train_eps', 'valid_u', 'valid_eps',
            'mmap', 'force_float32',
        )

        w = fno_width
        m = fno_modes

        # lift: 2 channels (u + eps_map) -> width
        self.lift = nn.Conv2d(2, w, kernel_size=1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([FourierBlock(w, m, m) for _ in range(fno_layers)])

        # project: width -> 1
        self.proj = nn.Sequential(
            nn.Conv2d(w, w, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w, 1, kernel_size=1),
        )

    def forward(self, x, epsilon=None):
        # x: (B,1,H,W), epsilon: (B,1) or (B,)
        B, _, H, W = x.shape
        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.view(-1, 1)

        eps_map = epsilon.view(B, 1, 1, 1).expand(B, 1, H, W)
        h = self.lift(torch.cat([x, eps_map], dim=1))  # (B, w, H, W)

        for layer in self.fourier_layers:
            h = layer(h)

        return self.proj(h)  # (B, 1, H, W)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("FNO_Epsilon", "FNO baseline options")
        group.add_argument('--fno_width', type=int, default=32)
        group.add_argument('--fno_modes', type=int, default=12)
        group.add_argument('--fno_layers', type=int, default=4)
        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None)
        data.add_argument('--train_eps', type=str, default=None)
        data.add_argument('--valid_u', type=str, default=None)
        data.add_argument('--valid_eps', type=str, default=None)
        data.add_argument('--mmap', action='store_true')
        data.add_argument('--force_float32', action='store_true')
        group.set_defaults(**{**get_default_args(FNO_Epsilon), **defaults})
        return parser


# =========================================================================
#  Variant 2: FNO + FiLM conditioning on epsilon  (stronger baseline)
# =========================================================================

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta, where gamma/beta come from eps."""
    def __init__(self, width, eps_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, eps_dim),
            nn.GELU(),
            nn.Linear(eps_dim, 2 * width),  # gamma & beta
        )
        # init to identity: gamma=1, beta=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[:width] = 1.0  # gamma init = 1

    def forward(self, x, epsilon):
        # x: (B, C, H, W), epsilon: (B, 1)
        params = self.mlp(epsilon)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class FourierBlockFiLM(nn.Module):
    """Fourier block with FiLM modulation after spectral+local path."""
    def __init__(self, width, modes1, modes2, eps_dim=64):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, kernel_size=1)
        self.film = FiLMBlock(width, eps_dim)

    def forward(self, x, epsilon):
        h = self.spectral(x) + self.conv(x)
        h = self.film(h, epsilon)
        return F.gelu(h)


class FNO_Epsilon_FiLM(MeanCurvatureEpsilonProblem):
    """
    FNO with FiLM-based epsilon conditioning at every Fourier layer.
    Stronger baseline: epsilon modulates features, not just appended to input.
    """
    def __init__(self,
                 fno_width=32,
                 fno_modes=12,
                 fno_layers=4,
                 eps_dim=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(
            'fno_width', 'fno_modes', 'fno_layers', 'eps_dim',
            'train_u', 'train_eps', 'valid_u', 'valid_eps',
            'mmap', 'force_float32',
        )

        w = fno_width
        m = fno_modes

        self.lift = nn.Conv2d(1, w, kernel_size=1)  # only u as input

        self.fourier_layers = nn.ModuleList([
            FourierBlockFiLM(w, m, m, eps_dim) for _ in range(fno_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(w, w, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(w, 1, kernel_size=1),
        )

    def forward(self, x, epsilon=None):
        B = x.shape[0]
        if epsilon is None:
            epsilon = x.new_full((B, 1), float(self.hparams.epsilon))
        if epsilon.dim() == 1:
            epsilon = epsilon.view(-1, 1)

        h = self.lift(x)
        for layer in self.fourier_layers:
            h = layer(h, epsilon)
        return self.proj(h)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("FNO_Epsilon_FiLM", "FNO-FiLM baseline options")
        group.add_argument('--fno_width', type=int, default=32)
        group.add_argument('--fno_modes', type=int, default=12)
        group.add_argument('--fno_layers', type=int, default=4)
        group.add_argument('--eps_dim', type=int, default=64)
        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None)
        data.add_argument('--train_eps', type=str, default=None)
        data.add_argument('--valid_u', type=str, default=None)
        data.add_argument('--valid_eps', type=str, default=None)
        data.add_argument('--mmap', action='store_true')
        data.add_argument('--force_float32', action='store_true')
        group.set_defaults(**{**get_default_args(FNO_Epsilon_FiLM), **defaults})
        return parser