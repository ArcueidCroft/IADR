#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
... run_dr1_bootstrap.sh ... master log... BOOT ... xlsx...

... (model_id, shape, eps_id) ...
  - PAPER_METRICS: L2_Tinf / V_Tinf / AIW_t  (mean / boot_std / ci_lo / ci_hi)
  - ... BOOT ...: NRMSE / RMSE / SSIM / MSE / MAE / L2_error / Vol_error / IW_B_abs / IW_E_abs / AIW_abs
  - timing: step_time_per_batch_(mean|median), step_time_per_sample_(mean|median)
  - ...: bootstrap N, T_eval, mode, AIW frame_idx
"""

import argparse
import re
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


START_RE = re.compile(
    r"Processing:\s*([A-Za-z0-9_-]+)\s+on\s+([A-Za-z0-9_]+?)_(\d+)\s*$",
    re.M,
)
BOOT_HDR_RE = re.compile(r"^=== Bootstrap \(mode=([a-z_]+),\s*B=(\d+),\s*N=(\d+),\s*T_eval=(\d+)\)", re.M)
AIW_FRAME_RE = re.compile(r"^\[INFO\] AIW single-frame:\s*rel_t=(\d+)\s+\(abs frame=(\d+),\s*ratio=([0-9.]+)\)", re.M)

# BOOT METRIC: mean=... boot_std=... CI95=[lo, hi]  [red=...]
_num_pat = r"(?:[0-9.eE+\-]+|NA)"
BOOT_LINE_RE = re.compile(
    r"^BOOT\s+([A-Za-z0-9_]+)\s*:\s*"
    r"mean=(" + _num_pat + r")\s+"
    r"boot_std=(" + _num_pat + r")\s+"
    r"CI95=\[\s*(" + _num_pat + r")\s*,\s*(" + _num_pat + r")\s*\]"
    r"(?:\s*\[red=([A-Za-z0-9_]+)(?:,\s*t=(\d+))?[^\]]*\])?",
    re.M,
)

STEP_TIME_BATCH_MEAN_RE   = re.compile(r"^step_time_per_batch_mean:\s*([0-9.eE+-]+)\s*$",   re.M)
STEP_TIME_BATCH_MEDIAN_RE = re.compile(r"^step_time_per_batch_median:\s*([0-9.eE+-]+)\s*$", re.M)
STEP_TIME_SAMPLE_MEAN_RE   = re.compile(r"^step_time_per_sample_mean:\s*([0-9.eE+-]+)\s*$",   re.M)
STEP_TIME_SAMPLE_MEDIAN_RE = re.compile(r"^step_time_per_sample_median:\s*([0-9.eE+-]+)\s*$", re.M)

SEED_SUFFIX_RE = re.compile(r"^(.*)_s(\d+)$")

PAPER_METRICS = ["L2_Tinf", "V_Tinf", "AIW_t"]


def autosize(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v = cell.value
            if v is None:
                continue
            s = f"{v:.3e}" if isinstance(v, float) else str(v)
            max_len = max(max_len, len(s))
        ws.column_dimensions[col_letter].width = min(38, max(10, max_len + 2))


def safe_sheet_title(t):
    t = re.sub(r"[:\\/?*\[\]]", "_", t).strip()
    return (t or "sheet")[:31]


def split_model_seed(model_id):
    m = SEED_SUFFIX_RE.match(model_id)
    if m:
        return m.group(1), int(m.group(2))
    return model_id, None


def parse_log(text):
    """... list of dict: model_id, model_base, seed, shape, eps_id, metrics{...}, ci_lo{...}, ci_hi{...}, boot_std{...}, ..."""
    rows = []
    matches = list(START_RE.finditer(text))
    for i, m in enumerate(matches):
        model_id, shape, eps_id = m.group(1), m.group(2), m.group(3)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        window = text[start:end]

        mode = boot_B = N_eval = T_eval = None
        h = BOOT_HDR_RE.search(window)
        if h:
            mode = h.group(1)
            boot_B = int(h.group(2))
            N_eval = int(h.group(3))
            T_eval = int(h.group(4))

        aiw_rel_t = aiw_abs_t = aiw_ratio = None
        af = AIW_FRAME_RE.search(window)
        if af:
            aiw_rel_t = int(af.group(1))
            aiw_abs_t = int(af.group(2))
            aiw_ratio = float(af.group(3))

        def _n(s):
            if s is None or s == "NA": return None
            return float(s)

        boot_metrics = {}  # key -> dict(mean, boot_std, ci_lo, ci_hi, red, t)
        for bm in BOOT_LINE_RE.finditer(window):
            k = bm.group(1)
            boot_metrics[k] = dict(
                mean=_n(bm.group(2)),
                boot_std=_n(bm.group(3)),
                ci_lo=_n(bm.group(4)),
                ci_hi=_n(bm.group(5)),
                red=bm.group(6) or "",
                t=int(bm.group(7)) if bm.group(7) else None,
            )

        def _f(rgx):
            mm = rgx.search(window)
            return float(mm.group(1)) if mm else None

        timing = dict(
            step_time_per_batch_mean=_f(STEP_TIME_BATCH_MEAN_RE),
            step_time_per_batch_median=_f(STEP_TIME_BATCH_MEDIAN_RE),
            step_time_per_sample_mean=_f(STEP_TIME_SAMPLE_MEAN_RE),
            step_time_per_sample_median=_f(STEP_TIME_SAMPLE_MEDIAN_RE),
        )

        if not boot_metrics:
            continue

        model_base, seed = split_model_seed(model_id)
        rows.append(dict(
            model_id=model_id, model_base=model_base, seed=seed,
            shape=shape, eps_id=eps_id,
            mode=mode, boot_B=boot_B, N_eval=N_eval, T_eval=T_eval,
            aiw_rel_t=aiw_rel_t, aiw_abs_t=aiw_abs_t, aiw_ratio=aiw_ratio,
            boot=boot_metrics, timing=timing,
        ))
    return rows


def build_xlsx(rows, out_path):
    wb = Workbook()
    wb.remove(wb.active)

    all_metric_keys = []
    seen = set()
    for r in rows:
        for k in r["boot"].keys():
            if k not in seen:
                seen.add(k); all_metric_keys.append(k)
    paper_first = [k for k in PAPER_METRICS if k in seen]
    other_keys = [k for k in all_metric_keys if k not in PAPER_METRICS]
    ordered_keys = paper_first + other_keys

    timing_cols = [
        "step_time_per_batch_mean", "step_time_per_batch_median",
        "step_time_per_sample_mean", "step_time_per_sample_median",
    ]

    ws = wb.create_sheet(title="raw")
    base_hdr = ["model_id", "model_base", "seed", "shape", "eps_id",
                "mode", "boot_B", "N_eval", "T_eval",
                "aiw_rel_t", "aiw_abs_t", "aiw_ratio"]
    metric_hdr = []
    for k in ordered_keys:
        metric_hdr += [f"{k}_mean", f"{k}_boot_std", f"{k}_ci_lo", f"{k}_ci_hi"]
    ws.append(base_hdr + metric_hdr + timing_cols)
    for r in rows:
        line = [r["model_id"], r["model_base"], r["seed"] if r["seed"] is not None else "",
                r["shape"], r["eps_id"], r["mode"], r["boot_B"], r["N_eval"], r["T_eval"],
                r["aiw_rel_t"], r["aiw_abs_t"], r["aiw_ratio"]]
        for k in ordered_keys:
            d = r["boot"].get(k)
            if d is None:
                line += [None, None, None, None]
            else:
                line += [d["mean"], d["boot_std"], d["ci_lo"], d["ci_hi"]]
        for tc in timing_cols:
            line.append(r["timing"].get(tc))
        ws.append(line)
    autosize(ws)

    for k in ordered_keys:
        ws = wb.create_sheet(title=safe_sheet_title(f"m_{k}"))
        ws.append(["model_id", "shape", "eps_id", "mean", "boot_std", "ci_lo", "ci_hi", "red", "t"])
        sub = []
        for r in rows:
            d = r["boot"].get(k)
            if d is None:
                continue
            sub.append((r["model_id"], r["shape"], r["eps_id"], d))
        sub.sort(key=lambda x: (x[0], x[1], x[2]))
        for mid, sh, eid, d in sub:
            ws.append([mid, sh, eid, d["mean"], d["boot_std"], d["ci_lo"], d["ci_hi"], d["red"], d["t"]])
        autosize(ws)

    ws = wb.create_sheet(title="paper_summary")
    hdr = ["model_id", "shape", "eps_id"]
    for k in paper_first:
        hdr += [f"{k}_mean", f"{k}_boot_std", f"{k}_ci_lo", f"{k}_ci_hi"]
    hdr += timing_cols
    ws.append(hdr)
    rs = sorted(rows, key=lambda r: (r["model_id"], r["shape"], r["eps_id"]))
    for r in rs:
        line = [r["model_id"], r["shape"], r["eps_id"]]
        for k in paper_first:
            d = r["boot"].get(k)
            if d is None:
                line += [None, None, None, None]
            else:
                line += [d["mean"], d["boot_std"], d["ci_lo"], d["ci_hi"]]
        for tc in timing_cols:
            line.append(r["timing"].get(tc))
        ws.append(line)
    autosize(ws)

    ws = wb.create_sheet(title="paper_meanstd_text")
    hdr = ["model_id", "shape", "eps_id"] + [f"{k}_mean±std" for k in paper_first]
    ws.append(hdr)
    for r in rs:
        line = [r["model_id"], r["shape"], r["eps_id"]]
        for k in paper_first:
            d = r["boot"].get(k)
            if d is None or d["mean"] is None:
                line.append("NA")
            elif d["boot_std"] is None:
                line.append(f"{d['mean']:.3e} (no std)")
            else:
                line.append(f"{d['mean']:.3e} ± {d['boot_std']:.2e}")
        ws.append(line)
    autosize(ws)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main():
    ap = argparse.ArgumentParser("bootstrap log -> xlsx")
    ap.add_argument("--log", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    print(f"[INFO] Reading log: {log_path}")
    rows = parse_log(text)
    if not rows:
        print("[ERROR] ... BOOT ... log ... 'Processing: <model> on <shape>_<eps_id>' ... 'BOOT <metric>:' ...")
        return

    bases = sorted({r["model_base"] for r in rows})
    shapes = sorted({r["shape"] for r in rows})
    print(f"[INFO] Total entries: {len(rows)}")
    print(f"[INFO] Models: {bases}")
    print(f"[INFO] Shapes: {shapes}")
    seen_metrics = set()
    for r in rows:
        seen_metrics |= set(r["boot"].keys())
    print(f"[INFO] Metrics: {sorted(seen_metrics)}")

    print(f"[INFO] Saving xlsx -> {out_path}")
    build_xlsx(rows, out_path)
    print("[OK] Done.")


if __name__ == "__main__":
    main()
