#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Allen-Cahn circle dataset with per-sample randomised radius and center.

Same LieSplitting solver as get_datasets_fs_random.py for consistency.
Shape knobs randomised per sample:
    radius      : uniform in [radius_min, radius_max]
    center_x/y  : uniform in [center_x_range, center_y_range]

Output (one file per epsilon):
    data : (1, T, N, 1, H, W)   -- N_eps=1 per file, matching per-eps split in sh
    eps  : (1,)
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


def parse_epsilon_list(s: str):
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def _sample_uniform(lo, hi):
    if lo == hi:
        return float(lo)
    return float(torch.empty(1).uniform_(float(lo), float(hi)).item())


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomised circle Allen-Cahn dataset (per-sample radius + center)"
    )

    parser.add_argument("--grid",        type=int,   default=256)
    parser.add_argument("--steps",       type=int,   default=199,
                        help="演化步数；T=steps+1 帧。用 699 可得 700 帧（与原 circle 数据集一致，但文件约 18 GB/eps/100 sample）")
    parser.add_argument("--num-samples", type=int,   default=100)
    parser.add_argument("--dt",          type=float, default=None)
    parser.add_argument("--save-dir",    type=str,   default="./std_test_datasets/circle")

    parser.add_argument("--epsilon-list", type=str,  default="0.0078",
                        help="逗号分隔的 epsilon 列表；sh 里每次只传一个")

    # 圆半径随机范围
    parser.add_argument("--radius-min",  type=float, default=0.25)
    parser.add_argument("--radius-max",  type=float, default=0.35)

    # 圆心随机范围（域 [0,1]^2 内；0.5 = 中心）
    parser.add_argument("--center-x-range", type=float, nargs=2, default=[0.47, 0.53],
                        metavar=("LO", "HI"))
    parser.add_argument("--center-y-range", type=float, nargs=2, default=[0.47, 0.53],
                        metavar=("LO", "HI"))

    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--prefix", type=str, default="circle_dataset")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d  = Domain([[0, 1], [0, 1]], args.grid)
    dx = 1.0 / args.grid
    dt = (2.0 * dx) ** 2 if (args.dt is None) else float(args.dt)
    H = W = args.grid
    N     = args.num_samples
    T     = args.steps + 1

    cx_lo, cx_hi = args.center_x_range
    cy_lo, cy_hi = args.center_y_range

    epsilon_set = parse_epsilon_list(args.epsilon_list)

    print(f"Generating circle dataset: {len(epsilon_set)} Eps × {N} Samples × {T} Frames")
    print(f"Grid: {H}×{W}, dx: {dx:.6g}, dt: {dt:.6g}, seed: {args.seed}")
    print(f"radius   : [{args.radius_min}, {args.radius_max}]")
    print(f"center_x : [{cx_lo}, {cx_hi}]")
    print(f"center_y : [{cy_lo}, {cy_hi}]")
    est_gb = len(epsilon_set) * N * T * 1 * H * W * 4 / 1e9
    print(f"Estimated total size: {est_gb:.2f} GB")

    for ei, eps in enumerate(epsilon_set):
        solver = LieSplitting(d, dt=dt, epsilon=eps)

        seq_samples = []
        pbar = tqdm(range(N), desc=f"eps[{ei+1}/{len(epsilon_set)}]={eps:.6g}", dynamic_ncols=True)

        for si in pbar:
            r  = _sample_uniform(args.radius_min, args.radius_max)
            cx = _sample_uniform(cx_lo, cx_hi)
            cy = _sample_uniform(cy_lo, cy_hi)

            s0 = shapes.translation(shapes.sphere(radius=r), shift=(cx, cy))
            u0 = AllenCahnProblem.profil(s0(*d.X), eps)

            frames = torch.empty((T, 1, H, W), dtype=u0.dtype)
            u = u0.clone()
            frames[0, 0] = u
            for t in range(1, T):
                u = solver(u)
                frames[t, 0] = u

            seq_samples.append(frames.unsqueeze(0))  # (1, T, 1, H, W)

        # (N, T, 1, H, W) -> (T, N, 1, H, W) -> (1, T, N, 1, H, W)
        seq_eps = torch.cat(seq_samples, dim=0)
        seq_eps = seq_eps.permute(1, 0, 2, 3, 4).contiguous().unsqueeze(0)

        data_path = os.path.join(args.save_dir, f"{args.prefix}.npy")
        eps_path  = os.path.join(args.save_dir, f"{args.prefix}_eps.npy")

        np.save(data_path, seq_eps.cpu().numpy())
        np.save(eps_path,  np.asarray([eps], dtype=np.float32))

        print(f"[eps={eps:.6g}] saved: {data_path}  shape={tuple(seq_eps.shape)}")

    print("\n=== DONE ===")
    print(f"grid={args.grid}, dx={dx:.6g}, dt={dt:.6g}")


if __name__ == "__main__":
    main()
