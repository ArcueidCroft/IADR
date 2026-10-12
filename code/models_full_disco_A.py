"""Full DISCO, variant A (DataModule-supplied T-frame context).

This is a faithful port of the DISCO architecture (Morel, Han, Oyallon,
ICML 2025; https://github.com/RudyMorel/DISCO) to the phase-field MCF task.
Unlike ``models_disco_unet.UNet_DISCO`` (which is DISCO's operator network
conditioned on the scalar epsilon through a tiny MLP), this file keeps the
ENTIRE DISCO stack:

    (1) T-frame short trajectory -> Transformer hypernetwork (Encoder +
        SpaceTimeBlock x processor_blocks) -> physics latent vector
        theta_latent in R^{embed_dim}
    (2) theta_latent -> param_gen_{common,adim,dim,channels} -> operator
        parameters theta, split by DISCO's weight masks
    (3) signed_sigmoid rescaling with per-parameter theta_norm
    (4) operator parameters drive the same DISCO U-Net, integrated from
        t=0 to t=1 with torchdiffeq's odeint_adjoint (bosh3 solver)

Two design decisions that define "full DISCO" here (per user confirmation
2026-04-20): (Q2) the model is NOT conditioned on epsilon -- it must
discover the dynamics from the T-frame context alone; (Q3) the single-step
prediction is a continuous-time ODE integration, not a discrete one-shot
forward. Q4: hyperparameters are the DISCO paper defaults (embed_dim=384,
patch_size=16, processor_blocks=12, num_heads=6, n_past=16).

Variant A (this file): the T-frame context is supplied by the DataModule.
Concretely, training_step expects a batch whose first entry has shape
(B, T, 1, H, W) with T == self.hparams.n_past. If the DataModule hands back
a 4-D (B, 1, H, W) tensor instead, this variant raises NotImplementedError
pointing the user at ``models_full_disco_B`` (which synthesises the context
by repeating the current frame).

Environment: requires ``timm`` (via DISCO's attention module) and ``einops``.
Also relies on the DISCO repo being checked out at ``DISCO_ROOT`` (default
``~/study/code/DISCO``); override via the ``DISCO_ROOT`` environment
variable. The repo ships its own ``torchdiffeq`` and ``standardize`` under
``src/``; those are imported here unchanged.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

# ---------------------------------------------------------------------
# Import DISCO's own modules (faithful reproduction requires the real code)
# ---------------------------------------------------------------------
_DISCO_ROOT = os.environ.get("DISCO_ROOT", str(Path.home() / "study/code/DISCO"))
if _DISCO_ROOT not in sys.path:
    sys.path.insert(0, _DISCO_ROOT)

try:
    from src.models.disco import Hypernetwork, OperatorNetwork, functional_call_batched
    from src.utils.database import standardize
    from src.torchdiffeq import odeint_adjoint
except ImportError as e:
    raise ImportError(
        f"Could not import DISCO modules from DISCO_ROOT={_DISCO_ROOT}. "
        f"Set the DISCO_ROOT environment variable to your DISCO checkout, "
        f"and ensure `timm` and `einops` are installed. Underlying error: {e}"
    )

from nnpf.problems import AllenCahnProblem as Problem
from nnpf.problems.mean_curvature import MeanCurvatureEpsilonProblem
from nnpf.utils import get_default_args


# ---------------------------------------------------------------------
# DISCO defaults (Q4 -- user confirmed to keep paper defaults)
# ---------------------------------------------------------------------

_DISCO_DEFAULTS = dict(
    n_past=16,                 # T (context length)
    patch_size=16,
    embed_dim=384,
    hpnn_head_hidden_dim=384,
    processor_blocks=12,
    num_heads=6,
    bias_type="rel",
    drop_path=0.1,
    opnn_start=8,
    opnn_norm_groups=4,
    hpnn_groups=12,            # hard-coded to 12 in DISCO's src/models/__init__.py
    ode_max_steps=32,          # DISCO's bosh3 ODE max integration steps; renamed from
                               # `max_steps` to avoid PL Trainer's `--max_steps` (which
                               # is the GLOBAL training step limit and would otherwise
                               # be hijacked via dest aliasing).
    rtol=5e-6,
    norm_factor=2.0,           # signed_sigmoid factor
)


class FullDisco_A(MeanCurvatureEpsilonProblem):
    """Full DISCO architecture, variant A: DataModule supplies T-frame context.

    Training / validation batch contract:
        batch[0]         : context, shape (B, T, 1, H, W), T == self.hparams.n_past
        batch[1]         : epsilon (ignored -- full DISCO is epsilon-agnostic)
        batch[2 : 2+K]   : K future targets, each (B, 1, H, W)

    If batch[0] has fewer than 5 dimensions, the DataModule has not been
    upgraded to produce context windows; switch to FullDisco_B instead, which
    uses frame-repetition to synthesise the T-frame context internally.
    """

    def __init__(
        self,
        # ODE / solver
        ode_max_steps: int = _DISCO_DEFAULTS["ode_max_steps"],
        rtol: float = _DISCO_DEFAULTS["rtol"],
        norm_factor: float = _DISCO_DEFAULTS["norm_factor"],
        # hypernetwork (DISCO Transformer)
        n_past: int = _DISCO_DEFAULTS["n_past"],
        patch_size: int = _DISCO_DEFAULTS["patch_size"],
        embed_dim: int = _DISCO_DEFAULTS["embed_dim"],
        hpnn_head_hidden_dim: int = _DISCO_DEFAULTS["hpnn_head_hidden_dim"],
        processor_blocks: int = _DISCO_DEFAULTS["processor_blocks"],
        num_heads: int = _DISCO_DEFAULTS["num_heads"],
        bias_type: str = _DISCO_DEFAULTS["bias_type"],
        drop_path: float = _DISCO_DEFAULTS["drop_path"],
        hpnn_groups: int = _DISCO_DEFAULTS["hpnn_groups"],
        # operator network (DISCO U-Net)
        opnn_start: int = _DISCO_DEFAULTS["opnn_start"],
        opnn_norm_groups: int = _DISCO_DEFAULTS["opnn_norm_groups"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters(
            "ode_max_steps", "rtol", "norm_factor",
            "n_past", "patch_size", "embed_dim", "hpnn_head_hidden_dim",
            "processor_blocks", "num_heads", "bias_type", "drop_path",
            "hpnn_groups",
            "opnn_start", "opnn_norm_groups",
            "train_u", "train_eps", "valid_u", "valid_eps",
            "mmap", "force_float32",
        )

        # -------------------------------------------------------------
        # Operator network: DISCO's U-Net with periodic BC, 1 input/output
        # channel. Its parameters are *structure only* -- they will be
        # replaced sample-by-sample by the hypernet's output.
        # -------------------------------------------------------------
        self.opnn = OperatorNetwork(
            in_channels=1,
            start=int(opnn_start),
            spatial_ndims=2,
            boundary_conditions="periodic",
            norm_groups=int(opnn_norm_groups),
        )
        for p in self.opnn.parameters():
            p.requires_grad = False

        # Param layout + DISCO's three-way mask (adim / dim / channels).
        self._param_info = [(n, p.shape, p.numel()) for n, p in self.opnn.named_parameters()]
        self.numel_opnn_params = sum(numel for _, _, numel in self._param_info)
        self.register_buffer("param_mask", self._make_param_mask(self.opnn))

        self.n_adim = int((self.param_mask == 0).sum().item())
        self.n_dim = int((self.param_mask == 1).sum().item())
        self.n_channels = int((self.param_mask == 2).sum().item())

        print(
            f"[FullDisco_A] operator U-Net params: {self.numel_opnn_params} "
            f"(adim={self.n_adim}, dim={self.n_dim}, channels={self.n_channels})"
        )

        # -------------------------------------------------------------
        # Per-parameter theta_norm (|max| at init time), used for DISCO's
        # signed_sigmoid rescaling. Matches DISCO.init_normalizations.
        # -------------------------------------------------------------
        self.register_buffer("theta_norm", self._init_theta_norm(self.opnn))

        # -------------------------------------------------------------
        # Hypernetwork: DISCO's Encoder + SpaceTimeBlock x processor_blocks.
        # ndims=[2], n_states=1 specialises it to 2-D single-channel data.
        # -------------------------------------------------------------
        self.hpnn = Hypernetwork(
            patch_size=int(patch_size),
            n_states=1,
            ndims=[2],
            embed_dim=int(embed_dim),
            groups=int(hpnn_groups),
            processor_blocks=int(processor_blocks),
            drop_path=float(drop_path),
            num_heads=int(num_heads),
            bias_type=str(bias_type),
        )

        # Parameter generator: theta_latent -> operator params (split by mask).
        self.param_gen_common = nn.Sequential(
            nn.Linear(int(embed_dim), int(hpnn_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hpnn_head_hidden_dim), int(hpnn_head_hidden_dim)),
            nn.GELU(),
        )
        self.param_gen_adim = nn.Linear(int(hpnn_head_hidden_dim), self.n_adim)
        self.param_gen_dim = nn.Linear(int(hpnn_head_hidden_dim), self.n_dim)
        self.param_gen_channels = nn.Linear(int(hpnn_head_hidden_dim), self.n_channels)

        # Static channel label (we only have one physical field, the phase u).
        self.register_buffer("state_labels", torch.arange(1, dtype=torch.long))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FullDisco_A] trainable parameters: {trainable}")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _make_param_mask(opnn) -> torch.Tensor:
        """DISCO's three-way weight mask (adim=0, dim=1, channels=2).

        Mirrors src/models/disco.py::DISCO.make_weights_masks verbatim.
        """
        params = dict(opnn.named_parameters())
        mask = {k: torch.ones_like(v, dtype=torch.int32) for k, v in params.items()}
        for i, (k, v) in enumerate(params.items()):
            if v.shape[-1] == 1:  # 1x1 conv = fully-connected -> adim
                mask[k].fill_(0)
            if i == 0 or i == len(params) - 1:  # adapter_in / adapter_out
                mask[k].fill_(2)
        return parameters_to_vector(mask.values())

    @staticmethod
    def _init_theta_norm(opnn) -> torch.Tensor:
        params = dict(opnn.named_parameters())
        params_max = {k: torch.full_like(v, v.abs().max().item()) for k, v in params.items()}
        return parameters_to_vector(params_max.values())

    @staticmethod
    def signed_sigmoid(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
        return factor * (2 * torch.sigmoid(2 * x / factor) - 1)

    # -----------------------------------------------------------------
    # Core forward: (B, T, 1, H, W) -> (B, 1, H, W)
    # -----------------------------------------------------------------

    def _forward_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """DISCO forward: T-frame context -> one predicted next frame.

        Faithfully mirrors src/models/disco.py::DISCO.forward (without the
        multi-dataset adapters, since we have exactly one 2-D channel).
        """
        B, T, C, H, W = context.shape
        assert C == 1, f"FullDisco_A expects a single channel (phase u), got C={C}"
        assert T == int(self.hparams.n_past), (
            f"context length T={T} does not match hparams.n_past={self.hparams.n_past}"
        )

        # Standardise over (time + space) per (batch, channel). Matches
        # DISCO forward lines 477-480.
        spatial_dims = tuple(range(3, context.ndim))  # (3, 4) for 2D
        x, mean_all, std_all = standardize(context, dims=(1, *spatial_dims), return_stats=True)

        x_input = x[:, -1, ...]  # (B, C, H, W)

        # Transformer hypernet on the T-frame context.
        theta_latent = self.hpnn(x, self.state_labels)  # (B, T, E, H', W', D)
        theta_latent = theta_latent.mean((1, 3, 4, 5))  # -> (B, E)

        # Parameter generator (three masked branches).
        z = self.param_gen_common(theta_latent)
        theta_adim = self.param_gen_adim(z)
        theta_dim = self.param_gen_dim(z)
        theta_ch = self.param_gen_channels(z)

        theta = torch.zeros(B, self.numel_opnn_params, device=x.device, dtype=x.dtype)
        theta[:, self.param_mask == 0] = theta_adim
        theta[:, self.param_mask == 1] = theta_dim
        theta[:, self.param_mask == 2] = theta_ch

        # DISCO signed_sigmoid normalisation (disco.py:500-504).
        tn = self.theta_norm.to(theta.dtype)
        theta = theta * tn / tn.pow(2.0).mean().pow(0.5)
        theta = tn * self.signed_sigmoid(theta / tn, factor=float(self.hparams.norm_factor))

        # Standardise the ODE initial state again on its spatial dims only.
        spatial_dims_xi = tuple(range(2, x_input.ndim))
        x_input, mean_t, std_t = standardize(x_input, dims=spatial_dims_xi, return_stats=True)

        # ODE integrator (bosh3, adjoint method) from t=0 to t=1.
        def f(t, y):
            y = y.unsqueeze(1)  # (B, 1, C, H, W) -> vmap over batch dim 0
            y = functional_call_batched(self.opnn, theta, y)
            return y.squeeze(1)

        options = {"min_step": 1.0 / float(self.hparams.ode_max_steps)}
        t = torch.linspace(0.0, 1.0, 2, device=x.device)
        _, y = odeint_adjoint(
            f,
            x_input,
            t=t,
            rtol=float(self.hparams.rtol),
            method="bosh3",
            options=options,
            adjoint_params=(theta,),
        )

        # De-standardise back to physical scale.
        y_pred = y[-1]                           # (B, C, H, W)
        y_pred = y_pred * std_t + mean_t
        y_pred = y_pred.unsqueeze(1)             # (B, 1, C, H, W)
        y_pred = y_pred * std_all + mean_all     # broadcast over time dim
        y_pred = y_pred.squeeze(1)               # (B, C, H, W)

        return y_pred

    # -----------------------------------------------------------------
    # Public forward (matches the DataModule contract of variant A).
    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
        """Run one DISCO prediction step.

        Args:
            x: (B, T, 1, H, W) T-frame context.
            epsilon: accepted for signature compatibility but ignored.

        Returns:
            (B, 1, H, W) predicted next frame.
        """
        if x.dim() != 5:
            raise NotImplementedError(
                f"FullDisco_A requires a 5-D (B, T, 1, H, W) context from the "
                f"DataModule, got x.dim()={x.dim()} with shape {tuple(x.shape)}. "
                f"Either upgrade the DataModule to produce T-frame context "
                f"windows, or use FullDisco_B which synthesises the context "
                f"by repeating the current frame."
            )
        return self._forward_from_context(x)

    # -----------------------------------------------------------------
    # Training / validation loops
    # -----------------------------------------------------------------

    def _rollout_loss(self, batch):
        context, _eps, *targets = batch
        if context.dim() != 5:
            raise NotImplementedError(
                f"FullDisco_A.training_step expects batch[0] to be a 5-D "
                f"(B, T, 1, H, W) context. Got {context.dim()}-D tensor "
                f"with shape {tuple(context.shape)}. Upgrade the DataModule "
                f"or switch to FullDisco_B."
            )
        loss = context.new_zeros([])
        for target in targets:
            out = self._forward_from_context(context)                    # (B, 1, H, W)
            loss = loss + self.loss(out, target)
            # Sliding window: drop the oldest frame, append the prediction.
            context = torch.cat([context[:, 1:], out.unsqueeze(1)], dim=1)
        return loss / max(len(targets), 1)

    def training_step(self, batch, batch_idx):
        loss = self._rollout_loss(batch)
        return self.dispatch_metrics({"loss": loss})

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._rollout_loss(batch)}

    # -----------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        g = parser.add_argument_group(
            "FullDisco_A",
            "Faithful DISCO port (variant A: DataModule supplies T-frame context).",
        )
        # ODE / solver
        g.add_argument("--ode_max_steps", type=int,
                       default=_DISCO_DEFAULTS["ode_max_steps"])
        g.add_argument("--rtol", type=float, default=_DISCO_DEFAULTS["rtol"])
        g.add_argument("--norm_factor", type=float, default=_DISCO_DEFAULTS["norm_factor"])
        # hypernetwork
        g.add_argument("--n_past", type=int, default=_DISCO_DEFAULTS["n_past"])
        g.add_argument("--patch_size", type=int, default=_DISCO_DEFAULTS["patch_size"])
        g.add_argument("--embed_dim", type=int, default=_DISCO_DEFAULTS["embed_dim"])
        g.add_argument("--hpnn_head_hidden_dim", type=int,
                       default=_DISCO_DEFAULTS["hpnn_head_hidden_dim"])
        g.add_argument("--processor_blocks", type=int,
                       default=_DISCO_DEFAULTS["processor_blocks"])
        g.add_argument("--num_heads", type=int, default=_DISCO_DEFAULTS["num_heads"])
        g.add_argument("--bias_type", type=str, default=_DISCO_DEFAULTS["bias_type"])
        g.add_argument("--drop_path", type=float, default=_DISCO_DEFAULTS["drop_path"])
        g.add_argument("--hpnn_groups", type=int, default=_DISCO_DEFAULTS["hpnn_groups"])
        # operator network
        g.add_argument("--opnn_start", type=int, default=_DISCO_DEFAULTS["opnn_start"])
        g.add_argument("--opnn_norm_groups", type=int,
                       default=_DISCO_DEFAULTS["opnn_norm_groups"])

        d = parser.add_argument_group("Data", "Dataset paths")
        d.add_argument("--train_u", type=str, default=None)
        d.add_argument("--train_eps", type=str, default=None)
        d.add_argument("--valid_u", type=str, default=None)
        d.add_argument("--valid_eps", type=str, default=None)
        d.add_argument("--mmap", action="store_true")
        d.add_argument("--force_float32", action="store_true")

        g.set_defaults(**{**get_default_args(FullDisco_A), **defaults})
        return parser


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--B", type=int, default=2)
    args = ap.parse_args()

    # A minimal instantiation. MeanCurvatureEpsilonProblem may demand
    # additional init kwargs depending on the local nnpf version; pass
    # whatever is required via **defaults here.
    model = FullDisco_A(
        max_epochs=1, batch_size=args.B, lr=1e-4,
    )
    T = model.hparams.n_past
    x = torch.randn(args.B, T, 1, args.H, args.H)
    with torch.no_grad():
        y = model(x)
    print(f"FullDisco_A output: {y.shape}")
    assert y.shape == (args.B, 1, args.H, args.H)
