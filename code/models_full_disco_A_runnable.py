"""Runnable wrapper around FullDisco_A.

FullDisco_A expects batch[0] to be a (B, T_past, 1, H, W) history window, but
the default DataModule of nnpf's MeanCurvatureEpsilonProblem only produces
(B, 1, H, W) single frames, so plain FullDisco_A would raise
NotImplementedError on the first call to _rollout_loss.

This file:
  1. subclasses FullDisco_A as FullDiscoA_Run, inheriting the architecture,
     forward, and loss logic unchanged;
  2. overrides train_dataloader / val_dataloader to read mmap-backed
     (E, T, N, 1, H, W) npy files via a sliding window of size T_past + K,
     yielding (context, eps, target_1, ..., target_K) tuples that match
     FullDisco_A's contract.

Assumptions:
  - npy shapes are (E, T_total, N, 1, H, W), the same layout as the IADR
    training tensors;
  - eps npy shapes are (E,) or (E, 1);
  - K = self.hparams.train_steps (read from the yml; default 1).

If nnpf injects a trainer-level DataModule that overrides the module-level
hook, this layer is bypassed; in that case the first run's log will show
single-frame batches and one must patch nnpf or add a separate fit script.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models_full_disco_A import FullDisco_A
from nnpf.utils import get_default_args


def _load_eps_array(eps_path: str, n_eps_expected: int) -> np.ndarray:
    e = np.load(eps_path).astype(np.float32).squeeze()
    if e.ndim == 0:
        e = np.array([float(e)], dtype=np.float32)
    if e.ndim == 1:
        e = e.reshape(-1)
    assert e.shape[0] == n_eps_expected, (
        f"eps npy shape {e.shape} does not match data E={n_eps_expected}"
    )
    return e


class _ContextWindowDataset(Dataset):
    """Slide a (T_past + K)-frame window across the time axis of an
    (E, T, N, 1, H, W) npy. Each item is
        (context: (T_past, 1, H, W) float32,
         eps:     scalar float32,
         target_0: (1, H, W) float32, ..., target_{K-1}: (1, H, W) float32)
    """

    def __init__(self, data_path: str, eps_path: str, T_past: int, K: int,
                 mmap: bool = True, force_float32: bool = False):
        super().__init__()
        mode = "r" if mmap else None
        self.arr = np.load(data_path, mmap_mode=mode)
        if self.arr.ndim != 6 or self.arr.shape[3] != 1:
            raise ValueError(
                f"expected (E, T, N, 1, H, W), got {self.arr.shape}")
        self.E, self.T, self.N, _, self.H, self.W = self.arr.shape
        self.T_past = int(T_past)
        self.K = int(K)
        self.window = self.T_past + self.K
        if self.T < self.window:
            raise ValueError(
                f"trajectory length T={self.T} < T_past({self.T_past}) + K({self.K})")
        self.num_starts = self.T - self.window + 1     # valid t_start count per (e, n)
        self.eps = _load_eps_array(eps_path, self.E)
        self.force_float32 = bool(force_float32)
        self._dtype = np.float32 if self.force_float32 else self.arr.dtype

    def __len__(self):
        return self.E * self.N * self.num_starts

    def __getitem__(self, idx):
        starts_per_e = self.N * self.num_starts
        e = idx // starts_per_e
        rem = idx % starts_per_e
        n = rem // self.num_starts
        t0 = rem % self.num_starts

        ctx_np = self.arr[e, t0 : t0 + self.T_past, n]            # (T_past, 1, H, W)
        if ctx_np.dtype != np.float32:
            ctx_np = ctx_np.astype(np.float32)
        else:
            ctx_np = np.ascontiguousarray(ctx_np)
        ctx = torch.from_numpy(ctx_np)

        targets = []
        for k in range(self.K):
            t = self.arr[e, t0 + self.T_past + k, n]               # (1, H, W)
            if t.dtype != np.float32:
                t = t.astype(np.float32)
            else:
                t = np.ascontiguousarray(t)
            targets.append(torch.from_numpy(t))

        eps_val = torch.tensor(float(self.eps[e]), dtype=torch.float32)
        return (ctx, eps_val, *targets)


class FullDiscoA_Run(FullDisco_A):
    """FullDisco_A + sliding-window dataloaders so it can train on existing
    (E, T, N, 1, H, W) npy without modifying nnpf's DataModule infra."""

    @property
    def example_input_array(self):
        N = self.hparams.N
        H, W = (int(N), int(N)) if isinstance(N, int) else (int(N[0]), int(N[-1]))
        return torch.zeros(1, int(self.hparams.n_past), 1, H, W)

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = FullDisco_A.add_model_specific_args(parent_parser, defaults)
        try:
            parser.add_argument("--train_steps", type=int, default=1,
                                help="multipoint unrolling K (number of target frames per sample)")
            parser.add_argument("--num_workers", type=int, default=0)
        except Exception:
            pass
        parser.set_defaults(**{**get_default_args(FullDiscoA_Run), **defaults})
        return parser

    def _build_dataset(self, data_path: str, eps_path: str) -> _ContextWindowDataset:
        T_past = int(self.hparams.n_past)
        K = int(getattr(self.hparams, "train_steps", 1))
        mmap = bool(getattr(self.hparams, "mmap", True))
        ff32 = bool(getattr(self.hparams, "force_float32", False))
        ds = _ContextWindowDataset(data_path, eps_path,
                                   T_past=T_past, K=K,
                                   mmap=mmap, force_float32=ff32)
        return ds

    def _common_loader(self, ds, shuffle: bool) -> DataLoader:
        bs = int(self.hparams.batch_size)
        nw = int(getattr(self.hparams, "num_workers", 0))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=True,
            drop_last=shuffle,                          # train drops last partial; val keeps it
        )

    def train_dataloader(self):
        if not getattr(self.hparams, "train_u", None):
            raise ValueError("train_u not set in yml")
        if not getattr(self.hparams, "train_eps", None):
            raise ValueError("train_eps not set in yml")
        ds = self._build_dataset(self.hparams.train_u, self.hparams.train_eps)
        print(f"[FullDiscoA_Run] train dataset: E={ds.E} T={ds.T} N={ds.N} "
              f"window=T_past({ds.T_past})+K({ds.K})  num_starts={ds.num_starts}  "
              f"len={len(ds)}")
        return self._common_loader(ds, shuffle=True)

    def val_dataloader(self):
        if not getattr(self.hparams, "valid_u", None):
            raise ValueError("valid_u not set in yml")
        if not getattr(self.hparams, "valid_eps", None):
            raise ValueError("valid_eps not set in yml")
        ds = self._build_dataset(self.hparams.valid_u, self.hparams.valid_eps)
        print(f"[FullDiscoA_Run] val dataset: E={ds.E} T={ds.T} N={ds.N} "
              f"window=T_past({ds.T_past})+K({ds.K})  num_starts={ds.num_starts}  "
              f"len={len(ds)}")
        return self._common_loader(ds, shuffle=False)


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-u",    required=True)
    ap.add_argument("--train-eps",  required=True)
    ap.add_argument("--T-past",     type=int, default=16)
    ap.add_argument("--K",          type=int, default=1)
    args = ap.parse_args()
    ds = _ContextWindowDataset(args.train_u, args.train_eps,
                               T_past=args.T_past, K=args.K)
    print(f"dataset len: {len(ds)}, E={ds.E} T={ds.T} N={ds.N}")
    item = ds[0]
    print(f"item shapes: ctx={tuple(item[0].shape)}, "
          f"eps={item[1].item()}, targets={[tuple(t.shape) for t in item[2:]]}")
