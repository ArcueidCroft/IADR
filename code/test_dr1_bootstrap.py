#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DR1 rollout test + sample bootstrap on per-sample metrics

import os
import sys
import time
import argparse
from pathlib import Path
import inspect

import numpy as np
import torch


def _stage(msg):
    """... + flush ...,... log file ..."""
    ts = time.strftime("%H:%M:%S")
    print(f"[STAGE {ts}] {msg}", flush=True)
    sys.stdout.flush()
import torch.nn.functional as F
from tqdm import tqdm

from nnpf.problems import Problem

torch.set_float32_matmul_precision("high")


def resolve_ckpt_path(p: str, prefer: str = "last") -> str:
    pth = Path(p)
    if pth.is_file():
        return str(pth)
    ckpt_dir = pth / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"... checkpoints ...{ckpt_dir}")
    last = ckpt_dir / "last.ckpt"

    def _epoch_step_key(p):
        import re
        m = re.search(r"epoch=(\d+)-step=(\d+)", p.name)
        if not m:
            return (-1, -1)
        return (int(m.group(1)), int(m.group(2)))

    epoch_cands = sorted(ckpt_dir.glob("epoch=*-step=*.ckpt"),
                         key=_epoch_step_key, reverse=True)
    if prefer == "last":
        if last.exists():
            return str(last)
        raise FileNotFoundError(f"--ckpt_prefer=last ... {last} ...")
    if prefer == "epoch":
        if epoch_cands:
            return str(epoch_cands[0])
        raise FileNotFoundError(f"--ckpt_prefer=epoch ... {ckpt_dir} ... epoch=*-step=*.ckpt")
    if prefer == "mtime":
        cands = sorted(ckpt_dir.glob("*.ckpt"), key=lambda s: s.stat().st_mtime, reverse=True)
        if cands:
            return str(cands[0])
        raise FileNotFoundError(f"checkpoints ... .ckpt...{ckpt_dir}")
    if epoch_cands:
        return str(epoch_cands[0])
    best_cands = sorted(ckpt_dir.glob("best*.ckpt"),
                        key=lambda s: s.stat().st_mtime, reverse=True)
    if best_cands:
        return str(best_cands[0])
    if last.exists():
        return str(last)
    cands = sorted(ckpt_dir.glob("*.ckpt"), key=lambda s: s.stat().st_mtime, reverse=True)
    if cands:
        return str(cands[0])
    raise FileNotFoundError(f"... ckpt...{p}")


def _ensure_all_cross_modules(model, device, dtype):
    cnt = 0
    for m in model.modules():
        if hasattr(m, "_ensure_cross") and callable(getattr(m, "_ensure_cross")):
            try:
                m._ensure_cross(device=device, dtype=dtype)
                cnt += 1
            except TypeError:
                pass
    return cnt


def load_model_with_cross(ckpt_path, device, dtype):
    _stage(f"load_model_with_cross: Problem.load_from_checkpoint('{ckpt_path}')")
    model = Problem.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    _stage("load_model_with_cross: model loaded; .eval().to(device,dtype)")
    model.eval().to(device=device, dtype=dtype)
    _stage("load_model_with_cross: torch.load(ckpt) for state_dict scan")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    has_cross = any(".cross." in k for k in sd.keys())
    if has_cross:
        _stage("load_model_with_cross: detected .cross.* weights; _ensure_cross + reload")
        print("[INFO] ckpt ... *.cross.* ... cross ...", flush=True)
        n = _ensure_all_cross_modules(model, device=device, dtype=dtype)
        print(f"[INFO] ... _ensure_cross ...: {n}", flush=True)
        model.load_state_dict(sd, strict=False)
        _stage("load_model_with_cross: state_dict reloaded")
    _stage("load_model_with_cross: DONE")
    return model


class LazySeq:
    """... mmap ... RAM...
    ... seq[slice, t_int] -> (B,1,H,W) float32 CPU tensor
         seq[int,  t_int, 0] -> (H,W) float32 numpy  (... vis)
    """
    def __init__(self, arr, meta):
        self._arr = arr
        self._is_grouped = meta["is_grouped"]
        if self._is_grouped:
            N_eps, T, N_per, _, H, W = arr.shape
            self.shape = (N_eps * N_per, T, 1, H, W)
            self._N_per = N_per
        else:
            N, T, H, W = arr.shape
            self.shape = (N, T, 1, H, W)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 3:
            n, t, _ = idx
            return self._fetch(slice(n, n + 1), t)[0, 0].numpy()  # (H,W)
        if isinstance(idx, tuple) and len(idx) == 2:
            sl, t = idx
            return self._fetch(sl, t)  # (B,1,H,W) tensor
        raise IndexError(f"LazySeq ... {idx!r}")

    def _fetch(self, sl, t):
        arr = self._arr
        start = sl.start if sl.start is not None else 0
        stop  = sl.stop
        if self._is_grouped:
            N_per = self._N_per
            chunks = []
            i = start
            while i < stop:
                ei       = i // N_per
                pi_start = i % N_per
                pi_end   = min(N_per, pi_start + (stop - i))
                chunk = np.asarray(arr[ei, t, pi_start:pi_end, 0], dtype=np.float32)
                chunks.append(chunk)
                i += pi_end - pi_start
            frames = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        else:
            frames = np.asarray(arr[start:stop, t], dtype=np.float32)  # (B,H,W)
        return torch.from_numpy(np.ascontiguousarray(frames))[:, None]  # (B,1,H,W)


def load_seq_grouped(path, device, dtype):
    _stage(f"load_seq_grouped: np.load(mmap_mode='r') '{path}'")
    arr = np.load(path, mmap_mode='r')
    _stage(f"load_seq_grouped: shape={arr.shape}, dtype={arr.dtype} ... lazy mode, no upfront copy")
    if arr.ndim == 6 and arr.shape[3] == 1:
        N_eps, T, N_per, C1, H, W = arr.shape
        assert C1 == 1
        meta = dict(is_grouped=True, N_eps=N_eps, N_per=N_per, T=T, H=H, W=W)
        _stage(f"load_seq_grouped: DONE grouped lazy, N_total={N_eps*N_per}, T={T}, H={H}, W={W}")
        return LazySeq(arr, meta), meta
    if arr.ndim == 4:
        N, T, H, W = arr.shape
        meta = dict(is_grouped=False, N_eps=None, N_per=None, T=T, H=H, W=W)
        _stage(f"load_seq_grouped: DONE flat lazy, N={N}, T={T}, H={H}, W={W}")
        return LazySeq(arr, meta), meta
    raise ValueError(f"... test.npy ...{arr.shape}")


def build_eps_per_sample(eps_values_path, meta, device, dtype):
    e = np.load(eps_values_path).astype(np.float32).squeeze()
    if meta["is_grouped"]:
        N_eps, N_per = meta["N_eps"], meta["N_per"]
        if e.ndim == 0: e = np.full((N_eps, 1), float(e), dtype=np.float32)
        if e.ndim == 1: e = e.reshape(N_eps, 1)
        if e.shape != (N_eps, 1):
            raise ValueError(f"--eps_values ... (N_eps,) ... (N_eps,1)... {e.shape}")
        e_full = np.repeat(e, repeats=N_per, axis=0)
        return torch.from_numpy(e_full).to(device=device, dtype=dtype)
    if e.ndim == 0: e = np.array([float(e)], dtype=np.float32)
    if e.ndim == 1: e = e[:, None]
    return torch.from_numpy(e).to(device=device, dtype=dtype)




def _domain_dx_prod(L, H, W):
    return (L / float(H)) * (L / float(W))


def mass_dx(u_hw, dx_prod):
    return dx_prod * u_hw.sum(dim=(-2, -1))


def l2_error_dx(u, u_ref, dx_prod):
    if u.dim() == 4: u = u[:, 0]; u_ref = u_ref[:, 0]
    diff2 = (u - u_ref).pow(2).sum(dim=(-2, -1))
    return (dx_prod * diff2).sqrt()


def volume_error_dx(u, u_ref, dx_prod):
    if u.dim() == 4: u = u[:, 0]; u_ref = u_ref[:, 0]
    return (mass_dx(u, dx_prod) - mass_dx(u_ref, dx_prod)).abs()


def _grad_mag_center(u, dx, dy):
    if u.dim() == 4: u2 = u[:, 0]
    else: u2 = u
    up = F.pad(u2, (1, 1, 1, 1), mode="replicate")
    ux = (up[:, 1:-1, 2:] - up[:, 1:-1, :-2]) / (2.0 * dx)
    uy = (up[:, 2:, 1:-1] - up[:, :-2, 1:-1]) / (2.0 * dy)
    return torch.sqrt(ux * ux + uy * uy + 1e-30)


@torch.no_grad()
def avg_interface_width_AIW(u, dx_prod, dx, dy, low, high):
    if not (high > low):
        raise ValueError(f"AIW: require high>low, got low={low}, high={high}")
    if u.dim() == 3: u = u[:, None, :, :]
    band = ((u > low) & (u < high)).squeeze(1)
    area_band = dx_prod * band.sum(dim=(-2, -1)).to(u.dtype)
    g = _grad_mag_center(u, dx=dx, dy=dy)
    g_band = g * band.to(g.dtype)
    length_est = (dx_prod * g_band.sum(dim=(-2, -1))) / ((high - low) + 1e-30)
    return area_band / (length_est + 1e-30)


@torch.no_grad()
def compute_aiw_abs(pred, tgt, dx_prod, L, aiw_low, aiw_high):
    _, _, H, W = pred.shape
    dx = L / float(H); dy = L / float(W)
    aiw_p = avg_interface_width_AIW(pred, dx_prod=dx_prod, dx=dx, dy=dy, low=aiw_low, high=aiw_high)
    aiw_t = avg_interface_width_AIW(tgt, dx_prod=dx_prod, dx=dx, dy=dy, low=aiw_low, high=aiw_high)
    return (aiw_p - aiw_t).abs().detach().cpu().numpy()


def _supports_epsilon(model):
    try:
        sig = inspect.signature(model.forward)
        return ("epsilon" in sig.parameters) or ("**" in str(sig))
    except Exception:
        return False


def safe_forward(model, xb, eb=None, use_amp=False, initial_micro=0):
    device = xb.device; B = xb.shape[0]
    micro = initial_micro if initial_micro and initial_micro < B else B
    outs = []; use_eps = (eb is not None) and _supports_epsilon(model)
    with torch.inference_mode():
        while True:
            try:
                outs.clear()
                for s in range(0, B, micro):
                    xs = xb[s:s + micro]; es = None if eb is None else eb[s:s + micro]
                    if use_amp and device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            ys = model(x=xs, epsilon=es) if use_eps else model(x=xs)
                            if isinstance(ys, tuple): ys = ys[0]
                    else:
                        ys = model(x=xs, epsilon=es) if use_eps else model(x=xs)
                        if isinstance(ys, tuple): ys = ys[0]
                    outs.append(ys); del xs, es, ys
                    if device.type == "cuda": torch.cuda.empty_cache()
                out = torch.cat(outs, dim=0); outs.clear()
                if device.type == "cuda": torch.cuda.empty_cache()
                return out
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise
                if micro == 1: raise
                micro = max(1, micro // 2)
                if device.type == "cuda": torch.cuda.empty_cache()


def save_data_to_npy(path, data, meta=None, grouped_like_input=False):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)
    tensor = data.detach().cpu()
    if grouped_like_input and meta and meta.get("is_grouped", False):
        N_total, T, C1, H, W = tensor.shape
        N_eps, N_per = int(meta["N_eps"]), int(meta["N_per"])
        if N_eps * N_per != N_total:
            raise ValueError(f"N_total({N_total}) != N_eps({N_eps})*N_per({N_per})")
        tensor = tensor.reshape(N_eps, N_per, T, C1, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
    np.save(path, tensor.numpy())


def get_model_name_from_checkpoint_arg(checkpoint_arg):
    p = Path(checkpoint_arg)
    if p.is_dir(): return p.name
    if p.is_file():
        if p.parent.name == "checkpoints": return p.parent.parent.name
        return p.parent.name
    return Path(checkpoint_arg).name


def _make_rwb_138():
    """... plot_frames_per_sample.py ... RWB_138 colormap..."""
    from matplotlib.colors import LinearSegmentedColormap
    red_138  = (138 / 255.0, 0.0, 0.0)
    white    = (1.0, 1.0, 1.0)
    blue_138 = (0.0, 0.0, 138 / 255.0)
    return LinearSegmentedColormap.from_list(
        "RWB_138",
        [(0.0, blue_138), (0.5, white), (1.0, red_138)],
        N=256,
    )


def render_sample0_per_frame(frames_list, t_start, out_dir, dpi, stride=1):
    """frames_list: list of (H, W) np.float32, in time order starting at t = t_start.
    ... frame_<abs_t:04d>.png ... out_dir...RWB_138..."""
    import matplotlib.pyplot as plt
    arr = np.stack(frames_list, axis=0)            # (T_total, H, W)
    T_total, H, W = arr.shape
    sel = list(range(0, T_total, max(1, int(stride))))
    if sel and sel[-1] != T_total - 1:
        sel.append(T_total - 1)
    sub = arr[sel]
    vmin = float(sub.min()); vmax = float(sub.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6
    cmap = _make_rwb_138()
    os.makedirs(out_dir, exist_ok=True)
    print(f"[VIS] rendering {len(sel)} PNGs to {out_dir} | vmin={vmin:.4g} vmax={vmax:.4g}")
    for k, ti in enumerate(sel):
        img = arr[ti]
        abs_t = t_start + ti
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper")
        ax.set_axis_off()
        fig.savefig(os.path.join(out_dir, f"frame_{abs_t:04d}.png"), dpi=dpi, pad_inches=0)
        plt.close(fig)


PER_SAMPLE_KEYS = ["L2_error", "Vol_error", "AIW_abs"]

PAPER_METRICS = {
    "L2_Tinf": ("L2_error",  "max"),
    "V_Tinf":  ("Vol_error", "max"),
    "AIW_t":   ("AIW_abs",   "single_t"),
}


def bootstrap_sample(per_sample_vec, B, rng):
    """Sample bootstrap: per-sample ... (N,)... B ..."""
    N = per_sample_vec.shape[0]
    idx = rng.integers(0, N, size=(B, N))
    return per_sample_vec[idx].mean(axis=1)


def main():
    ap = argparse.ArgumentParser("DR1 rollout test + bootstrap std")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--ckpt_prefer", choices=["last", "epoch", "mtime", "best"], default="last")
    ap.add_argument("--test", required=True, type=str)
    ap.add_argument("--eps_values", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--microbatch", type=int, default=0)
    ap.add_argument("--timing_warmup", type=int, default=2)
    ap.add_argument("--save_rollout_preds", type=str, default="")
    ap.add_argument("--t_start", type=int, default=0)
    ap.add_argument("--t_end", type=int, default=-1)
    ap.add_argument("--force_train_mode", action="store_true", default=False)
    ap.add_argument("--domain_diameter", type=float, default=1.0)
    ap.add_argument("--aiw_low", type=float, default=0.02)
    ap.add_argument("--aiw_high", type=float, default=0.98)
    ap.add_argument("--aiw_single_ratio", type=float, default=0.8,
                    help="AIW_t ... rollout ...0.8 = 80%...")
    ap.add_argument("--aiw_single_t", type=int, default=-1,
                    help=">=0 ... ratio")
    ap.add_argument("--bootstrap_B", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument("--bootstrap_metrics", type=str,
                    default="L2_Tinf,V_Tinf,AIW_t",
                    help="... bootstrap ... 3 ... PAPER_METRICS ... PER_SAMPLE_KEYS... mean over T ...")
    ap.add_argument("--save_per_sample_metrics", type=str, default="",
                    help="... per-sample (N,) ... full (N,T_eval) ... npz")
    ap.add_argument("--vis_outdir", type=str, default="",
                    help="... sample-0 ... PNG ...")
    ap.add_argument("--vis_dpi", type=int, default=300)
    ap.add_argument("--vis_stride", type=int, default=1, help="...")
    args = ap.parse_args()

    _stage(f"main: argparse done, args.checkpoint={args.checkpoint!r}, args.test={args.test!r}")
    device = torch.device(args.device); dtype = torch.float32
    _stage(f"main: device={device}, dtype={dtype}, CUDA available={torch.cuda.is_available()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    ckpt_path = resolve_ckpt_path(args.checkpoint, prefer=args.ckpt_prefer)
    model_name = get_model_name_from_checkpoint_arg(args.checkpoint)
    _stage(f"main: resolved ckpt_path={ckpt_path}")
    print(f"[INFO] Using ckpt: {ckpt_path}", flush=True)
    print(f"[INFO] Model: {model_name}", flush=True)
    model = load_model_with_cross(ckpt_path, device, dtype)
    _stage("main: load_model_with_cross returned")
    if args.force_train_mode:
        print("[WARN] ... model.train()")
        model.train()
    else:
        model.eval()

    _stage(f"main: load_seq_grouped('{args.test}') ...")
    seq, meta = load_seq_grouped(args.test, device, dtype)
    _stage(f"main: seq tensor shape={tuple(seq.shape)}, meta={meta}")
    N_total, T, _, H, W = seq.shape
    t_start = int(args.t_start); t_end = int(args.t_end)
    if t_end < 0: t_end = T
    if not (0 <= t_start < T): raise ValueError("--t_start ...")
    if not (t_start < t_end <= T): raise ValueError("--t_end ...")
    eval_T = t_end - t_start; eval_steps = eval_T - 1
    if eval_steps <= 0: raise ValueError("...")
    print(f"[INFO] Eval range: t in [{t_start}, {t_end})  N={N_total} eval_steps={eval_steps}")

    _stage(f"main: build_eps_per_sample('{args.eps_values}')")
    eps_full = build_eps_per_sample(args.eps_values, meta, device, dtype) if args.eps_values else None
    _stage(f"main: eps_full shape={None if eps_full is None else tuple(eps_full.shape)}")
    if eps_full is not None and eps_full.shape[0] != N_total:
        raise ValueError(f"epsilon ... {eps_full.shape[0]} ≠ N_total {N_total}")

    L = float(args.domain_diameter)
    dx_prod = _domain_dx_prod(L, H, W)

    per_sample_acc = {k: np.full((N_total, eval_steps), np.nan, dtype=np.float32)
                      for k in PER_SAMPLE_KEYS}

    preds = torch.empty((N_total, T, 1, H, W), device=device, dtype=dtype) if args.save_rollout_preds else None
    Bsz = args.batch_size

    sample0_frames = []  # list of (H, W) numpy arrays in time order, starting from t_start
    if args.vis_outdir:
        sample0_frames.append(seq[0, t_start, 0].astype(np.float32))

    step_times = []; step_batch_sizes = []
    WARMUP_N = int(args.timing_warmup)

    with torch.inference_mode():
        for i in tqdm(range(0, N_total, Bsz), desc="Free-rollout"):
            sl = slice(i, min(i + Bsz, N_total))
            u = seq[sl, t_start].to(device=device, dtype=dtype)
            if preds is not None: preds[sl, t_start] = u

            for t in range(t_start, t_end - 1):
                rel_t = t - t_start
                eb = None if eps_full is None else eps_full[sl]
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                u_next = safe_forward(model, u, eb=eb, use_amp=args.amp, initial_micro=args.microbatch)
                if device.type == "cuda": torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
                step_batch_sizes.append(int(u.shape[0]))
                tgt = seq[sl, t + 1].to(device=device, dtype=dtype)

                l2b = l2_error_dx(u_next, tgt, dx_prod=dx_prod).detach().cpu().numpy().astype(np.float32)
                volb = volume_error_dx(u_next, tgt, dx_prod=dx_prod).detach().cpu().numpy().astype(np.float32)
                per_sample_acc["L2_error"][sl, rel_t] = l2b
                per_sample_acc["Vol_error"][sl, rel_t] = volb
                aiw_b = compute_aiw_abs(u_next, tgt, dx_prod=dx_prod, L=L,
                                        aiw_low=float(args.aiw_low), aiw_high=float(args.aiw_high))
                per_sample_acc["AIW_abs"][sl, rel_t] = aiw_b.astype(np.float32)

                if args.vis_outdir and sl.start == 0:
                    sample0_frames.append(u_next[0, 0].detach().cpu().numpy().astype(np.float32))

                if preds is not None: preds[sl, t + 1] = u_next
                u = u_next

    print("\n=== Metrics (mean over all steps & samples) ===")
    for src_key in ("L2_error", "Vol_error", "AIW_abs"):
        mat = per_sample_acc[src_key]
        if np.isnan(mat).all(): continue
        print(f"{src_key}: {float(np.nanmean(mat)):.6e}")

    # ---- timing ----
    n_calls_total = len(step_times)
    skip = min(WARMUP_N, max(0, n_calls_total - 1))
    timed_t = step_times[skip:]; timed_b = step_batch_sizes[skip:]
    print(f"\n=== Timing ===")
    print(f"step_calls_total: {n_calls_total}")
    print(f"step_calls_warmup: {skip}")
    if timed_t:
        per_batch_arr = np.asarray(timed_t, dtype=np.float64)
        per_sample_arr = per_batch_arr / np.asarray(timed_b, dtype=np.float64)
        print(f"step_time_per_batch_mean: {float(per_batch_arr.mean()):.6e}")
        print(f"step_time_per_batch_median: {float(np.median(per_batch_arr)):.6e}")
        print(f"step_time_per_sample_mean: {float(per_sample_arr.mean()):.6e}")
        print(f"step_time_per_sample_median: {float(np.median(per_sample_arr)):.6e}")

    # ---- bootstrap (sample only) ----
    rng = np.random.default_rng(args.bootstrap_seed)
    Bboot = int(args.bootstrap_B)

    if int(args.aiw_single_t) >= 0:
        aiw_rel_t = int(args.aiw_single_t) - t_start - 1
    else:
        aiw_rel_t = int(np.floor(float(args.aiw_single_ratio) * (eval_steps - 1)))
    aiw_rel_t = max(0, min(eval_steps - 1, aiw_rel_t))
    aiw_abs_t = t_start + 1 + aiw_rel_t

    print(f"\n=== Bootstrap (mode=sample, B={Bboot}, N={N_total}, T_eval={eval_steps}) ===")
    print(f"[INFO] AIW single-frame: rel_t={aiw_rel_t}  (abs frame={aiw_abs_t}, ratio={args.aiw_single_ratio})")
    if N_total < 2:
        print(f"[ERROR] N_total={N_total} < 2...sample bootstrap ... 2 ... N=100... bootstrap... mean...")

    def reduce_per_sample(metric_name):
        """... (per_sample_vec, reduction_tag)... None ..."""
        if metric_name in PAPER_METRICS:
            src_key, red = PAPER_METRICS[metric_name]
            if src_key not in per_sample_acc: return None
            mat = per_sample_acc[src_key]
            if np.isnan(mat).all(): return None
            if red == "max":
                vec = np.nanmax(mat, axis=1)
            elif red == "single_t":
                vec = mat[:, aiw_rel_t]
            else:
                vec = np.nanmean(mat, axis=1)
            return vec, red
        if metric_name not in per_sample_acc: return None
        mat = per_sample_acc[metric_name]
        if np.isnan(mat).all(): return None
        return np.nanmean(mat, axis=1), "mean"

    wanted = [m.strip() for m in args.bootstrap_metrics.split(",") if m.strip()]
    boot_summary = {}
    per_sample_means = {}
    for k in wanted:
        r = reduce_per_sample(k)
        if r is None:
            continue
        per_sample_vec, red = r
        per_sample_means[k] = per_sample_vec.astype(np.float32)
        m = float(np.nanmean(per_sample_vec))
        if N_total < 2:
            tag = f" [red={red}" + (f", t={aiw_rel_t}" if red == "single_t" else "") + ", N<2 -> no std]"
            print(f"BOOT {k}: mean={m:.6e}  boot_std=NA  CI95=[NA, NA]{tag}")
            boot_summary[k] = dict(mean=m, boot_std=None, ci_lo=None, ci_hi=None, red=red)
            continue
        try:
            boot_means = bootstrap_sample(per_sample_vec.astype(np.float64), Bboot, rng)
        except Exception as e:
            print(f"[WARN] bootstrap {k} ...{e}")
            continue
        s = float(np.nanstd(boot_means, ddof=1))
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
        boot_summary[k] = dict(mean=m, boot_std=s, ci_lo=lo, ci_hi=hi, red=red)
        tag = f" [red={red}" + (f", t={aiw_rel_t}" if red == "single_t" else "") + "]"
        print(f"BOOT {k}: mean={m:.6e}  boot_std={s:.6e}  CI95=[{lo:.6e}, {hi:.6e}]{tag}")

    if args.save_per_sample_metrics:
        save_dict = {}
        for k, v in per_sample_means.items():
            save_dict[f"persample_{k}"] = v
        for src_key in ("L2_error", "Vol_error", "AIW_abs"):
            mat = per_sample_acc[src_key]
            if not np.isnan(mat).all():
                save_dict[f"full_{src_key}"] = mat
        save_dict["_meta_N_eval_steps"] = np.array([N_total, eval_steps], dtype=np.int64)
        save_dict["_meta_bootstrap"] = np.array(
            [Bboot, args.bootstrap_seed], dtype=np.int64)
        save_dict["_meta_aiw_frame"] = np.array([aiw_rel_t, aiw_abs_t], dtype=np.int64)
        out_p = Path(args.save_per_sample_metrics)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_p, **save_dict)
        print(f"[OK] per-sample metrics saved -> {out_p}")

    if args.save_rollout_preds:
        save_data_to_npy(args.save_rollout_preds, preds, meta=meta, grouped_like_input=True)
        print(f"[OK] preds saved -> {args.save_rollout_preds}")

    if args.vis_outdir and sample0_frames:
        render_sample0_per_frame(
            sample0_frames, t_start=t_start,
            out_dir=args.vis_outdir, dpi=int(args.vis_dpi),
            stride=int(args.vis_stride),
        )


if __name__ == "__main__":
    main()
