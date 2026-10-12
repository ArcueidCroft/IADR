#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Allen-Cahn dataset with randomly-perturbed spiral-ripple initial shapes.

Same forward solver as the original get_datasets_sd.py (LieSplitting), but
each sample now draws 7 geometry knobs from user-specified ranges:
    base_radius, ripple_amplitude, ripple_r_freq, ripple_theta_freq,
    rotation_angle, center_x, center_y

Output:
    data: (N_eps, T, N, 1, H, W)
    eps : (N_eps,)
"""

import os
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm

import nnpf.shapes as shapes
from nnpf.domain import Domain
from nnpf.problems import AllenCahnProblem


class LieSplitting:
    def __init__(self, domain: Domain, dt: float, epsilon: float):
        self.domain = domain
        self.dt = dt / 10.0
        self.epsilon = float(epsilon)
        self.kernel = 1.0 / (1.0 + self.dt * 4.0 * math.pi**2 * sum(k**2 for k in domain.K))

    @torch.no_grad()
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        for _ in range(10):
            u = u - self.dt / (self.epsilon**2) * u * (1.0 - u) * (1.0 - 2.0 * u)
            u = self.domain.ifft(self.domain.fft(u) * self.kernel)
        return u


def build_spiral_shape(
    *,
    base_radius,
    ripple_amplitude,
    ripple_r_freq,
    ripple_theta_freq,
    rotation_angle,
    center_x,
    center_y,
):
    s0 = shapes.sphere(radius=base_radius)

    def spiral_ripple(*X):
        x, y = X
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return ripple_amplitude * torch.sin(ripple_r_freq * r + ripple_theta_freq * theta)

    s0 = shapes.displace(s0, spiral_ripple)
    s0 = shapes.rotate(s0, theta=rotation_angle)
    s0 = shapes.translation(s0, shift=(center_x, center_y))
    return s0


def parse_epsilon_list(s: str):
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def _resolve_range(rng_arg, scalar_arg):
    """rng_arg is None or [lo, hi]; scalar_arg is the legacy single-value default.
    Returns (lo, hi). If rng_arg is None, degenerate to (scalar, scalar)."""
    if rng_arg is None:
        return float(scalar_arg), float(scalar_arg)
    return float(rng_arg[0]), float(rng_arg[1])


def _sample_uniform(lo, hi):
    if lo == hi:
        return float(lo)
    return float(torch.empty(1).uniform_(float(lo), float(hi)).item())


def _sample_int_inclusive(lo, hi):
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if lo_i == hi_i:
        return lo_i
    return int(torch.randint(lo_i, hi_i + 1, (1,)).item())


def main():
    parser = argparse.ArgumentParser(description="Generate randomised spiral-disk Allen-Cahn dataset")

    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--save-dir", type=str, default="./data_spiral_liesplitting")

    parser.add_argument("--epsilon-list", type=str, default="0.0078125")

    # Legacy single-value knobs (kept for backward compatibility)
    parser.add_argument("--base-radius", type=float, default=0.3)
    parser.add_argument("--ripple-amplitude", type=float, default=0.08)
    parser.add_argument("--ripple-r-freq", type=float, default=10.0)
    parser.add_argument("--ripple-theta-freq", type=float, default=5.0)
    parser.add_argument("--rotation-angle", type=float, default=math.pi / 6.0)
    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.5)

    # New range knobs (each takes "lo hi"); if not given, fall back to legacy single value
    parser.add_argument("--base-radius-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--ripple-amplitude-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--ripple-r-freq-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="径向频率，整数；浮点输入会四舍五入")
    parser.add_argument("--ripple-theta-freq-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="角向频率，整数；浮点输入会四舍五入")
    parser.add_argument("--center-x-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--center-y-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-rotate", action="store_true",
                        help="开则 rotation_angle 在 [0, 2π] 均匀随机；关则用 --rotation-angle 固定值")

    parser.add_argument("--prefix", type=str, default="spiral_dataset")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = Domain([[0, 1], [0, 1]], args.grid)
    dx = 1.0 / args.grid
    dt = (2.0 * dx) ** 2 if (args.dt is None) else float(args.dt)

    epsilon_set = parse_epsilon_list(args.epsilon_list)
    N_eps = len(epsilon_set)
    N_samples = args.num_samples
    T = args.steps + 1
    H = W = args.grid

    base_r_range  = _resolve_range(args.base_radius_range,        args.base_radius)
    amp_range     = _resolve_range(args.ripple_amplitude_range,   args.ripple_amplitude)
    rfreq_range   = _resolve_range(args.ripple_r_freq_range,      args.ripple_r_freq)
    tfreq_range   = _resolve_range(args.ripple_theta_freq_range,  args.ripple_theta_freq)
    cx_range      = _resolve_range(args.center_x_range,           args.center_x)
    cy_range      = _resolve_range(args.center_y_range,           args.center_y)

    print(f"Generating dataset: {N_eps} Eps × {N_samples} Samples × {T} Frames")
    print(f"Grid: {H}×{W}, dt: {dt:.6g}, seed: {args.seed}")
    print(f"Random ranges:")
    print(f"  base_radius      : {base_r_range}")
    print(f"  ripple_amplitude : {amp_range}")
    print(f"  ripple_r_freq    : {rfreq_range} (int)")
    print(f"  ripple_theta_freq: {tfreq_range} (int)")
    print(f"  center_x         : {cx_range}")
    print(f"  center_y         : {cy_range}")
    print(f"  rotation_angle   : {'[0, 2π] uniform' if args.random_rotate else f'fixed {args.rotation_angle:.4f}'}")

    all_eps_data = []

    for ei, eps in enumerate(epsilon_set):
        solver = LieSplitting(d, dt=dt, epsilon=eps)

        seq_samples = []
        pbar = tqdm(range(N_samples), desc=f"Eps[{ei+1}/{N_eps}]={eps:.6g}", dynamic_ncols=True)

        for si in pbar:
            base_radius      = _sample_uniform(*base_r_range)
            ripple_amplitude = _sample_uniform(*amp_range)
            ripple_r_freq    = float(_sample_int_inclusive(*rfreq_range))
            ripple_theta_freq = float(_sample_int_inclusive(*tfreq_range))
            center_x         = _sample_uniform(*cx_range)
            center_y         = _sample_uniform(*cy_range)
            if args.random_rotate:
                rotation_angle = float(torch.rand(1).item()) * 2.0 * math.pi
            else:
                rotation_angle = float(args.rotation_angle)

            s0 = build_spiral_shape(
                base_radius=base_radius,
                ripple_amplitude=ripple_amplitude,
                ripple_r_freq=ripple_r_freq,
                ripple_theta_freq=ripple_theta_freq,
                rotation_angle=rotation_angle,
                center_x=center_x,
                center_y=center_y,
            )

            u0 = AllenCahnProblem.profil(s0(*d.X), eps)

            frames = torch.empty((T, 1, H, W), dtype=u0.dtype)
            u = u0.clone()
            frames[0, 0] = u
            for t in range(1, T):
                u = solver(u)
                frames[t, 0] = u

            seq_samples.append(frames.unsqueeze(0))

        seq_eps_tensor = torch.cat(seq_samples, dim=0)
        seq_eps_tensor = seq_eps_tensor.permute(1, 0, 2, 3, 4).contiguous().unsqueeze(0)
        all_eps_data.append(seq_eps_tensor)

    final_data = torch.cat(all_eps_data, dim=0)

    data_path = os.path.join(args.save_dir, f"{args.prefix}.npy")
    eps_path = os.path.join(args.save_dir, f"{args.prefix}_eps.npy")

    np.save(data_path, final_data.cpu().numpy())
    np.save(eps_path, np.asarray(epsilon_set, dtype=np.float32))

    print("\n=== Generation Complete ===")
    print(f"Data Shape: {final_data.shape}")
    print(f"Saved Data: {data_path}")
    print(f"Saved Eps : {eps_path}")


if __name__ == "__main__":
    main()
