#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Allen-Cahn dataset with 5-arm star initial shapes (Figure 12 style),
with per-sample randomised geometry knobs.

Same forward solver as the original get_datasets_fs.py (LieSplitting), but
each sample now draws shape knobs from user-specified ranges:
    n_arms, arm_polygon_sides, arm_outer_radius, arm_shift_x, arm_shift_y,
    center_disk_radius, global_shift_x, global_shift_y, global_rotation

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


def build_star_shape(
    *,
    n_arms,
    arm_polygon_sides,
    arm_outer_radius,
    arm_shift_x,
    arm_shift_y,
    center_disk_radius,
    global_shift_x,
    global_shift_y,
    global_rotation,
):
    arm = shapes.regular_polygon(arm_polygon_sides, outer_radius=arm_outer_radius)
    arm = shapes.translation(arm, shift=(arm_shift_x, arm_shift_y))

    arms = []
    for i in range(n_arms):
        theta = global_rotation + i * 2.0 * torch.pi / n_arms
        arms.append(shapes.rotate(arm, theta=theta))

    s0 = shapes.union(*arms)
    s0 = shapes.union(s0, shapes.sphere(radius=center_disk_radius))
    s0 = shapes.translation(s0, shift=(global_shift_x, global_shift_y))
    return s0


def parse_epsilon_list(s: str):
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def _resolve_range(rng_arg, scalar_arg):
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
    parser = argparse.ArgumentParser(description="Generate randomised 5-arm-star Allen-Cahn dataset")

    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--save-dir", type=str, default="./data_star_liesplitting")

    parser.add_argument("--epsilon-list", type=str, default="0.0078125")

    # Legacy single-value knobs (kept for backward compatibility)
    parser.add_argument("--n-arms", type=int, default=5)
    parser.add_argument("--arm-polygon-sides", type=int, default=3)
    parser.add_argument("--arm-outer-radius", type=float, default=0.2)
    parser.add_argument("--arm-shift-x", type=float, default=0.4)
    parser.add_argument("--arm-shift-y", type=float, default=0.0)
    parser.add_argument("--center-disk-radius", type=float, default=0.12)
    parser.add_argument("--global-shift-x", type=float, default=0.5)
    parser.add_argument("--global-shift-y", type=float, default=0.5)

    # New range knobs
    parser.add_argument("--n-arms-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="...")
    parser.add_argument("--arm-polygon-sides-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="...")
    parser.add_argument("--arm-outer-radius-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--arm-shift-x-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--arm-shift-y-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--center-disk-radius-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--global-shift-x-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--global-shift-y-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-rotate", action="store_true",
                        help="... global_rotation ... [0, 2π] ...")

    parser.add_argument("--prefix", type=str, default="star_dataset")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = Domain([[0, 1], [0, 1]], args.grid)
    dx = 1.0 / args.grid
    dt = (2.0 * dx) ** 2 if (args.dt is None) else float(args.dt)

    epsilon_set = parse_epsilon_list(args.epsilon_list)
    N_eps = len(epsilon_set)
    N = args.num_samples
    T = args.steps + 1
    H = W = args.grid

    n_arms_range          = _resolve_range(args.n_arms_range,             args.n_arms)
    arm_sides_range       = _resolve_range(args.arm_polygon_sides_range,  args.arm_polygon_sides)
    arm_outer_r_range     = _resolve_range(args.arm_outer_radius_range,   args.arm_outer_radius)
    arm_shift_x_range     = _resolve_range(args.arm_shift_x_range,        args.arm_shift_x)
    arm_shift_y_range     = _resolve_range(args.arm_shift_y_range,        args.arm_shift_y)
    center_disk_r_range   = _resolve_range(args.center_disk_radius_range, args.center_disk_radius)
    global_shift_x_range  = _resolve_range(args.global_shift_x_range,     args.global_shift_x)
    global_shift_y_range  = _resolve_range(args.global_shift_y_range,     args.global_shift_y)

    print(f"Generating dataset: {N_eps} Eps × {N} Samples × {T} Frames")
    print(f"Grid: {H}×{W}, dt: {dt:.6g}, seed: {args.seed}")
    print(f"Random ranges:")
    print(f"  n_arms             : {n_arms_range} (int)")
    print(f"  arm_polygon_sides  : {arm_sides_range} (int)")
    print(f"  arm_outer_radius   : {arm_outer_r_range}")
    print(f"  arm_shift_x        : {arm_shift_x_range}")
    print(f"  arm_shift_y        : {arm_shift_y_range}")
    print(f"  center_disk_radius : {center_disk_r_range}")
    print(f"  global_shift_x     : {global_shift_x_range}")
    print(f"  global_shift_y     : {global_shift_y_range}")
    print(f"  global_rotation    : {'[0, 2π] uniform' if args.random_rotate else 'fixed 0.0'}")

    all_eps = []

    for ei, eps in enumerate(epsilon_set):
        solver = LieSplitting(d, dt=dt, epsilon=eps)

        seq_samples = []
        pbar = tqdm(range(N), desc=f"eps[{ei+1}/{N_eps}]={eps:.6g}", dynamic_ncols=True)

        for si in pbar:
            n_arms             = _sample_int_inclusive(*n_arms_range)
            arm_polygon_sides  = _sample_int_inclusive(*arm_sides_range)
            arm_outer_radius   = _sample_uniform(*arm_outer_r_range)
            arm_shift_x        = _sample_uniform(*arm_shift_x_range)
            arm_shift_y        = _sample_uniform(*arm_shift_y_range)
            center_disk_radius = _sample_uniform(*center_disk_r_range)
            global_shift_x     = _sample_uniform(*global_shift_x_range)
            global_shift_y     = _sample_uniform(*global_shift_y_range)
            if args.random_rotate:
                global_rotation = float(torch.rand(1).item()) * 2.0 * math.pi
            else:
                global_rotation = 0.0

            s0 = build_star_shape(
                n_arms=n_arms,
                arm_polygon_sides=arm_polygon_sides,
                arm_outer_radius=arm_outer_radius,
                arm_shift_x=arm_shift_x,
                arm_shift_y=arm_shift_y,
                center_disk_radius=center_disk_radius,
                global_shift_x=global_shift_x,
                global_shift_y=global_shift_y,
                global_rotation=global_rotation,
            )

            u0 = AllenCahnProblem.profil(s0(*d.X), eps)

            frames = torch.empty((T, 1, H, W), dtype=u0.dtype)
            u = u0.clone()
            frames[0, 0] = u
            for t in range(1, T):
                u = solver(u)
                frames[t, 0] = u

            seq_samples.append(frames.unsqueeze(0))

        seq_eps = torch.cat(seq_samples, dim=0)
        seq_eps = seq_eps.permute(1, 0, 2, 3, 4).contiguous().unsqueeze(0)
        all_eps.append(seq_eps)

    data = torch.cat(all_eps, dim=0)

    data_path = os.path.join(args.save_dir, f"{args.prefix}.npy")
    eps_path = os.path.join(args.save_dir, f"{args.prefix}_eps.npy")

    np.save(data_path, data.cpu().numpy())
    np.save(eps_path, np.asarray(epsilon_set, dtype=np.float32))

    print("\n=== DONE ===")
    print(f"grid={args.grid}, dx={dx:.6g}, dt={dt:.6g}")
    print(f"data saved: {data_path}  shape={tuple(data.shape)}")
    print(f"eps  saved: {eps_path}   shape={(len(epsilon_set),)}")


if __name__ == "__main__":
    main()
