#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paired Allen--Cahn rollouts from dumbbell initial shapes.

The dumbbell is the union of two disks and a capsule joining their centres.
This makes ``ball_radius`` and ``neck_width`` directly interpretable, unlike
applying ``shapes.rounding`` to the complete union, which expands every part
of the geometry.

Output
------
data
    ``(N_eps, T, N_samples, 1, H, W)`` float32 NumPy array.
eps
    ``(N_eps,)`` float32 NumPy array.
visualisations
    For every epsilon and sample, five equally spaced frames are saved as a
    montage. Individual frame PNGs can also be saved.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import nnpf.shapes as shapes
from nnpf.domain import Domain
from nnpf.functional.phase_field import profil as phase_field_profile


class LieSplitting:
    """First-order reaction--diffusion Lie splitting with refined substeps."""

    def __init__(
        self,
        domain: Domain,
        outer_dt: float,
        epsilon: float,
        substeps: int = 10,
    ):
        if substeps < 1:
            raise ValueError(f"substeps must be positive, got {substeps}")

        self.domain = domain
        self.substeps = int(substeps)
        self.dt = float(outer_dt) / self.substeps
        self.epsilon = float(epsilon)
        self.kernel = 1.0 / (
            1.0
            + self.dt
            * 4.0
            * math.pi**2
            * sum(k**2 for k in domain.K)
        )

    @torch.no_grad()
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        for _ in range(self.substeps):
            u = u - (
                self.dt
                / self.epsilon**2
                * u
                * (1.0 - u)
                * (1.0 - 2.0 * u)
            )
            u = self.domain.ifft(self.domain.fft(u) * self.kernel)
        return u


def build_dumbbell_shape(
    *,
    ball_radius: float,
    half_center_distance: float,
    neck_width: float,
    global_rotation: float,
    center_x: float,
    center_y: float,
):
    """Construct a connected 2-D dumbbell with an explicit neck width."""

    left = (-half_center_distance, 0.0)
    right = (half_center_distance, 0.0)

    dumbbell = shapes.union(
        shapes.sphere(radius=ball_radius, center=left),
        shapes.sphere(radius=ball_radius, center=right),
        shapes.capsule(left, right, thickness=neck_width),
    )
    dumbbell = shapes.rotate(dumbbell, theta=global_rotation)
    return shapes.translation(dumbbell, shift=(center_x, center_y))


def parse_epsilon_list(value: str):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("epsilon-list must contain a value")
    return [float(item) for item in values]


def sample_with_jitter(rng, base: float, total_jitter: float) -> float:
    """Sample uniformly from base +/- total_jitter / 2."""

    return float(base + total_jitter * (rng.random() - 0.5))


def epsilon_tag(epsilon: float) -> str:
    return f"{epsilon:.10g}".replace("-", "m").replace(".", "_")


def equally_spaced_indices(num_frames: int, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError(f"vis-num-frames must be positive, got {count}")
    if count > num_frames:
        raise ValueError(
            f"Cannot select {count} distinct visualisation frames from {num_frames} frames"
        )
    return np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int64)


def save_sample_visualisation(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    output_dir: Path,
    sample_idx: int,
    epsilon: float,
    outer_dt: float,
    cmap: str,
    save_individual: bool,
):
    """Save one five-frame montage and, optionally, the five raw PNGs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = frames[frame_indices]

    fig, axes = plt.subplots(
        1,
        len(frame_indices),
        figsize=(3.0 * len(frame_indices), 3.0),
        constrained_layout=True,
    )
    if len(frame_indices) == 1:
        axes = [axes]

    for ax, image, frame_idx in zip(axes, selected, frame_indices):
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(f"frame={int(frame_idx)}\nt={frame_idx * outer_dt:.5f}")
        ax.axis("off")

    fig.suptitle(f"sample={sample_idx:03d}, epsilon={epsilon:.6g}")
    panel_path = output_dir / f"sample_{sample_idx:03d}_five_frames.png"
    fig.savefig(panel_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    if save_individual:
        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for image, frame_idx in zip(selected, frame_indices):
            plt.imsave(
                sample_dir / f"frame_{int(frame_idx):04d}.png",
                image,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
            )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def validate_geometry(params, domain_margin: float):
    for idx, item in enumerate(params):
        for key in ("ball_radius", "half_center_distance", "neck_width"):
            if item[key] <= 0.0:
                raise ValueError(f"sample {idx}: {key} must be positive, got {item[key]}")
        if item["neck_width"] > 2.0 * item["ball_radius"]:
            raise ValueError(
                f"sample {idx}: neck_width={item['neck_width']:.4g} exceeds "
                f"the disk diameter={2.0 * item['ball_radius']:.4g}"
            )

        radial_extent = item["half_center_distance"] + item["ball_radius"]
        available = min(
            item["center_x"],
            1.0 - item["center_x"],
            item["center_y"],
            1.0 - item["center_y"],
        )
        if available - radial_extent < domain_margin:
            raise ValueError(
                f"sample {idx}: shape is too close to the periodic boundary; "
                f"estimated margin={available - radial_extent:.4g}, "
                f"required margin={domain_margin:.4g}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired dumbbell Allen-Cahn rollout datasets"
    )

    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Number of rollout steps; the output contains steps+1 frames",
    )
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--epsilon-list", type=str, default="0.0078")
    parser.add_argument("--save-dir", type=str, default="./data_dumbbell")
    parser.add_argument("--prefix", type=str, default="dumbbell_dataset")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")

    # Geometry. Jitter values denote the total width of the sampling interval.
    parser.add_argument("--ball-radius", type=float, default=0.12)
    parser.add_argument("--ball-radius-jitter", type=float, default=0.02)
    parser.add_argument(
        "--half-center-distance",
        "--center-dist",
        dest="half_center_distance",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--half-center-distance-jitter",
        "--center-jitter",
        dest="half_center_distance_jitter",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--neck-width",
        "--bar-width",
        dest="neck_width",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--neck-width-jitter",
        "--bar-width-jitter",
        dest="neck_width_jitter",
        type=float,
        default=0.02,
    )
    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.5)
    parser.add_argument(
        "--position-jitter",
        type=float,
        default=0.04,
        help="Total uniform jitter width applied independently to x and y",
    )
    parser.add_argument("--random-rotate", action="store_true")
    parser.add_argument("--rotation-angle", type=float, default=0.0)
    parser.add_argument(
        "--min-boundary-margin",
        type=float,
        default=0.05,
        help="Minimum conservative radial margin to the periodic boundary",
    )

    # Visualisation.
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--vis-num-frames", type=int, default=5)
    parser.add_argument("--vis-cmap", type=str, default="RdBu_r")
    parser.add_argument("--vis-save-individual", action="store_true")

    args = parser.parse_args()

    if args.grid < 8:
        raise ValueError("grid must be at least 8")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    data_path = save_dir / f"{args.prefix}.npy"
    eps_path = save_dir / f"{args.prefix}_eps.npy"
    manifest_path = save_dir / f"{args.prefix}_geometry.json"

    existing = [path for path in (data_path, eps_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        paths = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Pass --overwrite to replace them:\n  " + paths
        )

    epsilon_set = parse_epsilon_list(args.epsilon_list)
    if any(epsilon <= 0.0 for epsilon in epsilon_set):
        raise ValueError(f"All epsilon values must be positive: {epsilon_set}")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    geometry_params = []
    for _ in range(args.num_samples):
        theta = (
            float(rng.uniform(0.0, math.pi))
            if args.random_rotate
            else float(args.rotation_angle)
        )
        geometry_params.append(
            {
                "ball_radius": sample_with_jitter(
                    rng, args.ball_radius, args.ball_radius_jitter
                ),
                "half_center_distance": sample_with_jitter(
                    rng,
                    args.half_center_distance,
                    args.half_center_distance_jitter,
                ),
                "neck_width": sample_with_jitter(
                    rng, args.neck_width, args.neck_width_jitter
                ),
                "center_x": sample_with_jitter(
                    rng, args.center_x, args.position_jitter
                ),
                "center_y": sample_with_jitter(
                    rng, args.center_y, args.position_jitter
                ),
                "rotation": theta,
            }
        )

    validate_geometry(geometry_params, args.min_boundary_margin)

    domain = Domain([[0, 1], [0, 1]], args.grid, device=device)
    dx = 1.0 / args.grid
    outer_dt = (2.0 * dx) ** 2 if args.dt is None else float(args.dt)
    total_frames = args.steps + 1
    vis_indices = equally_spaced_indices(total_frames, args.vis_num_frames)

    vis_root = (
        Path(args.vis_dir).expanduser().resolve()
        if args.vis_dir
        else save_dir / f"{args.prefix}_visualisations"
    )
    vis_root.mkdir(parents=True, exist_ok=True)

    shape = (
        len(epsilon_set),
        total_frames,
        args.num_samples,
        1,
        args.grid,
        args.grid,
    )
    data = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    estimated_gb = np.prod(shape, dtype=np.int64) * 4 / 1e9
    print(
        f"Generating {len(epsilon_set)} eps x {args.num_samples} samples "
        f"x {total_frames} frames"
    )
    print(
        f"grid={args.grid}x{args.grid}, dx={dx:.8g}, dt={outer_dt:.8g}, "
        f"substeps={args.substeps}, device={device}"
    )
    print(f"output={data_path}  shape={shape}  estimated_size={estimated_gb:.2f} GB")
    print(f"visualisation frame indices={vis_indices.tolist()}")

    for eps_idx, epsilon in enumerate(epsilon_set):
        solver = LieSplitting(
            domain,
            outer_dt=outer_dt,
            epsilon=epsilon,
            substeps=args.substeps,
        )
        eps_vis_dir = vis_root / f"eps_{epsilon_tag(epsilon)}"

        progress = tqdm(
            enumerate(geometry_params),
            total=args.num_samples,
            desc=f"epsilon={epsilon:.6g}",
            dynamic_ncols=True,
        )
        for sample_idx, params in progress:
            initial_shape = build_dumbbell_shape(
                ball_radius=params["ball_radius"],
                half_center_distance=params["half_center_distance"],
                neck_width=params["neck_width"],
                global_rotation=params["rotation"],
                center_x=params["center_x"],
                center_y=params["center_y"],
            )
            signed_distance = initial_shape(*domain.X)

            # Use the same phase-field profile as the paper and existing data.
            u = phase_field_profile(signed_distance, epsilon)
            frames = torch.empty(
                (total_frames, args.grid, args.grid),
                dtype=torch.float32,
                device=device,
            )
            frames[0].copy_(u.to(dtype=torch.float32))

            for frame_idx in range(1, total_frames):
                u = solver(u)
                frames[frame_idx].copy_(u.to(dtype=torch.float32))

            frames_np = frames.detach().cpu().numpy()
            if not np.isfinite(frames_np).all():
                raise FloatingPointError(
                    f"Non-finite values at epsilon={epsilon}, sample={sample_idx}"
                )

            data[eps_idx, :, sample_idx, 0] = frames_np
            save_sample_visualisation(
                frames_np,
                frame_indices=vis_indices,
                output_dir=eps_vis_dir,
                sample_idx=sample_idx,
                epsilon=epsilon,
                outer_dt=outer_dt,
                cmap=args.vis_cmap,
                save_individual=args.vis_save_individual,
            )
            del frames, frames_np, u, signed_distance

        data.flush()

    np.save(eps_path, np.asarray(epsilon_set, dtype=np.float32))
    manifest = {
        "data_path": str(data_path),
        "eps_path": str(eps_path),
        "data_shape": list(shape),
        "dtype": "float32",
        "grid": args.grid,
        "dx": dx,
        "outer_dt": outer_dt,
        "substeps": args.substeps,
        "steps": args.steps,
        "frame_count": total_frames,
        "epsilon_values": epsilon_set,
        "seed": args.seed,
        "phase_field_profile": "0.5 * (1 - tanh(distance / (2 * epsilon)))",
        "visualisation_frame_indices": vis_indices.tolist(),
        "geometry": geometry_params,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("Generation complete")
    print(f"data:       {data_path}")
    print(f"epsilon:    {eps_path}")
    print(f"geometry:   {manifest_path}")
    print(f"visuals:    {vis_root}")


if __name__ == "__main__":
    main()
