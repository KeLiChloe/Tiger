#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-step (decoupled) pipeline (GENERAL VERSION):

Supports files like:
  exp_d_10.pkl
  exp_K_5.pkl
  exp_delta_0p1.pkl

STEP 1) Build CSV from exp_*.pkl
  python analysis/plot_trends_auto/plot_general.py build-csv --dir exp_feb_2026/varying_d

STEP 2) Plot from CSV (no pkl needed)
  python analysis/plot_trends_auto/plot_general.py plot-csv --csv figures/curves_summary.csv

Notes:
- CSV schema is unified:
    param_name, param_value, comparator, mean_pct, err_pct, n, file
- Plot auto-sets title/xlabel based on param_name:
    d     -> Dimension
    K     -> Ground-truth number of clusters
    delta -> Magnitude of interaction effects
"""

import os
import re
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# Keys to ignore when extracting comparators
# -----------------------------
IGNORE_KEYS_DEFAULT = {
    "dast",
    "exp_params",
    "seed",
}

# Default outlier policy (per comparator)
DEFAULT_REMOVE_EXTREME = {
    "gmm-standard": True,
    "kmeans-standard": True,
    "mst": True,
    "clr-standard": True,
    "t_learner": True,
    "x_learner": True,
    "dr_learner": True,
    "s_learner": True,
    "policy_tree": True,
}

DEFAULT_COLORS = {
    "kmeans-standard": "#FFD22FAF",
    "gmm-standard":    "#006135AF",
    "clr-standard":    "#F59134AF",
    "mst":             "#937860AF",
    "t_learner":       "#6BC735AF",
    "s_learner":       "#1F5BFFAF",
    "x_learner":       "#FF3D3DAF",
    "dr_learner":      "#7A5CFFAF",
    "policy_tree":     "#333333AF",
}

LABEL_MAP = {
    "kmeans-standard": "K-Means",
    "gmm-standard":    "GMM",
    "clr-standard":    "CLR",
    "mst":             "MST",
    "t_learner":       "T-Learner",
    "s_learner":       "S-Learner",
    "x_learner":       "X-Learner",
    "dr_learner":      "DR-Learner",
    "policy_tree":     "Policy Tree",
}

# -----------------------------
# Experiment parameter -> labels
# -----------------------------
X_LABEL_MAP = {
    "d": "Dimension $d$",
    "K": "Ground-truth number of clusters ($K$)",
    "delta": r"Magnitude of interaction effects ($\delta$)",
}

TITLE_MAP = {
    "d": "DAST Advantage vs. Dimension",
    "K": "DAST Advantage vs. Number of Clusters",
    "delta": "DAST Advantage vs. Interaction Strength",
}


# -----------------------------
# Plot style (publication-ready)
# -----------------------------
def set_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,

        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,

        "figure.dpi": 120,
        "savefig.dpi": 600,
    })


# -----------------------------
# General experiment parameter extraction
# -----------------------------
def extract_experiment_param(data, filepath):
    """
    Parse parameter name/value from filename:
      exp_d_10.pkl
      exp_K_5.pkl
      exp_delta_0p1.pkl  -> delta=0.1

    Returns:
      (param_name: str, param_value: float)
    """
    base = os.path.basename(filepath)
    m = re.match(r"exp_([A-Za-z]+)_(.+)\.pkl$", base)
    if not m:
        raise ValueError(f"Filename not recognized: {base}")

    param_name = m.group(1)
    raw_val = m.group(2)

    if param_name == "delta":
        raw_val = raw_val.replace("p", ".")

    try:
        param_value = float(raw_val)
    except ValueError as e:
        raise ValueError(f"Cannot parse param_value from filename: {base}") from e

    return param_name, param_value


def extract_comparators(data, ignore_keys):
    """Comparator keys are list-valued runs, excluding ignore_keys."""
    comps = [k for k in data.keys() if k not in ignore_keys]
    comps = [k for k in comps if isinstance(data.get(k), list)]
    return comps


def compute_improvement_ratios(data, comparators, eps=1e-12):
    """
    ratio = (dast_profit - comp_profit) / abs(comp_profit)
    Returns dict comp -> np.array of ratios (not percent)
    """
    improvement = {}
    dast_runs = data.get("dast", None)
    if not isinstance(dast_runs, list):
        raise ValueError("Missing or invalid 'dast' list in data.")
    n = len(dast_runs)

    for comp in comparators:
        comp_runs = data.get(comp, None)
        if not isinstance(comp_runs, list) or len(comp_runs) != n:
            continue

        ratios = []
        for i in range(n):
            dast_profit = dast_runs[i].get("implementation_profits", None)
            comp_profit = comp_runs[i].get("implementation_profits", None)
            if dast_profit is None or comp_profit is None:
                continue

            dast_profit = float(dast_profit)
            comp_profit = float(comp_profit)
            if abs(comp_profit) < eps:
                continue

            ratios.append((dast_profit - comp_profit) / abs(comp_profit))

        improvement[comp] = np.asarray(ratios, dtype=float)

    return improvement


# -----------------------------
# Filtering + summarization
# -----------------------------
def filter_ratios(
    improvement_ratios,
    remove_extreme_map,
    method="trim",          # "trim" or "winsorize" or None
    trim_k=3,
    sigma_clip=False,
    sigma=3.0,
):
    """
    method:
      - "trim": drop k smallest and k largest
      - "winsorize": cap at k-th and (n-k-1)-th order stats
      - None: no trimming/winsor
    """
    filtered = {}
    for comp, arr in improvement_ratios.items():
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            filtered[comp] = x
            continue

        if remove_extreme_map.get(comp, False) and x.size > 2 * trim_k + 2:
            xs = np.sort(x)
            if method == "trim":
                xs = xs[trim_k: -trim_k]
            elif method == "winsorize":
                lo = xs[trim_k]
                hi = xs[-trim_k - 1]
                xs = np.clip(xs, lo, hi)
            x = xs

        if sigma_clip and x.size > 2:
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))
            if sd > 0:
                x = x[np.abs(x - mu) <= sigma * sd]

        filtered[comp] = x

    return filtered


def summarize_ratios(filtered_ratios):
    """
    Return dict: comp -> (mean_pct, err_pct, n)

    - mean_pct: mean(ratio) * 100
    - err_pct:
        "95% CI" half-width style used in your current script
        (kept consistent with your format).
    """
    out = {}
    for comp, arr in filtered_ratios.items():
        if arr is None or len(arr) == 0:
            out[comp] = (np.nan, np.nan, 0)
            continue

        ratios_pct = np.asarray(arr, dtype=float) * 100.0
        n = int(ratios_pct.size)
        mean = float(np.mean(ratios_pct))

        if n < 2:
            out[comp] = (mean, np.nan, n)
        else:
            se = float(stats.sem(ratios_pct, nan_policy="omit"))
            tcrit = float(stats.t.ppf(0.95, n - 1))
            out[comp] = (mean, se * tcrit, n)

    return out


# -----------------------------
# STEP 1: Build CSV from PKLs
# -----------------------------
def build_csv_from_dir(
    in_dir,
    out_csv,
    sigma_clip=False,
    sigma=3.0,
    extreme_method="trim",   # "trim"|"winsorize"|"none"
    trim_k=3,
):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "exp_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(in_dir, 'exp_*.pkl')}")

    remove_extreme_map = DEFAULT_REMOVE_EXTREME
    method = None if extreme_method == "none" else extreme_method

    records = []
    all_comps_seen = set()
    param_names_seen = set()

    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
            
        # ---- PRINT exp_params for each pkl ----
        exp_params = data.get("exp_params", None)
        print(f"\n=== {os.path.basename(fp)} ===")
        print(exp_params)

        param_name, param_value = extract_experiment_param(data, fp)
        param_names_seen.add(param_name)

        comps = extract_comparators(data, IGNORE_KEYS_DEFAULT)
        all_comps_seen.update(comps)

        ratios = compute_improvement_ratios(data, comps)
        filtered = filter_ratios(
            ratios,
            remove_extreme_map=remove_extreme_map,
            method=method,
            trim_k=trim_k,
            sigma_clip=sigma_clip,
            sigma=sigma,
        )
        summary = summarize_ratios(filtered)

        for comp, (mean_pct, err_pct, n) in summary.items():
            records.append({
                "param_name": param_name,
                "param_value": param_value,
                "comparator": comp,
                "mean_pct": mean_pct,
                "err_pct": err_pct,
                "n": n,
                "file": os.path.basename(fp),
            })

    df = pd.DataFrame(records).sort_values(["param_name", "comparator", "param_value"])
    df.to_csv(out_csv, index=False)

    return df, sorted(all_comps_seen), sorted(param_names_seen), len(files)


# -----------------------------
# STEP 2: Plot from CSV
# -----------------------------
def plot_from_csv(
    csv_path,
    out_fig,
    show_band=False,
    band_alpha=0.12,
):
    df = pd.read_csv(csv_path)

    required = {"param_name", "param_value", "comparator", "mean_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["param_value"] = pd.to_numeric(df["param_value"], errors="coerce")
    df["mean_pct"] = pd.to_numeric(df["mean_pct"], errors="coerce")
    if "err_pct" in df.columns:
        df["err_pct"] = pd.to_numeric(df["err_pct"], errors="coerce")
    else:
        df["err_pct"] = np.nan

    df = df.dropna(subset=["param_name", "param_value", "comparator", "mean_pct"]).copy()

    # If CSV contains multiple param_name, we plot one at a time (pick the first)
    param_name = str(df["param_name"].iloc[0])
    df = df[df["param_name"] == param_name].copy()

    xlabel = X_LABEL_MAP.get(param_name, param_name)
    title = TITLE_MAP.get(param_name, f"DAST Advantage vs. {param_name}")

    os.makedirs(os.path.dirname(out_fig) or ".", exist_ok=True)
    out_base = os.path.splitext(out_fig)[0]

    set_plot_style()

    # ---- Layout parameters (only tune these) ----
    AX_TOP = 0.82
    LEGEND_Y = AX_TOP + 0.03
    TITLE_Y = AX_TOP + 0.10

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=AX_TOP)

    fig.suptitle(title, y=TITLE_Y)

    ORDER = [
        "gmm-standard",
        "kmeans-standard",
        "mst", 
        "clr-standard", 
        # "t_learner", 
        # "s_learner", 
        # "x_learner", 
        # "dr_learner", 
        # "causal_forest",
        #"policy_tree",
    ]
    comps_present = list(df["comparator"].unique())
    comps = [c for c in ORDER if c in comps_present]
    if not comps:
        comps = sorted(comps_present)

    marker_map = {c: "o" for c in comps}

    for comp in comps:
        sub = df[df["comparator"] == comp].sort_values("param_value")
        x = sub["param_value"].to_numpy(dtype=float)
        y = sub["mean_pct"].to_numpy(dtype=float)

        label = LABEL_MAP.get(comp, comp)
        color = DEFAULT_COLORS.get(comp, None)
        mk = marker_map.get(comp, "o")

        ax.plot(
            x, y,
            marker=mk,
            label=label,
            color=color,
            markeredgewidth=0.8,
        )

        if show_band:
            e = sub["err_pct"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
            if np.any(ok):
                ax.fill_between(
                    x[ok],
                    (y - e)[ok],
                    (y + e)[ok],
                    alpha=band_alpha,
                    color=color,
                    linewidth=0,
                )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("DAST advantage over comparators (%)")

    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    # ax.minorticks_on()
    # ax.tick_params(which="minor", length=3, width=0.6)

    unique_x = sorted(df["param_value"].dropna().unique())

    MAX_TICKS = 20

    if len(unique_x) > 0 and np.all(np.isfinite(unique_x)):
        if len(unique_x) <= MAX_TICKS:
            ticks = unique_x
        else:
            idx = np.linspace(0, len(unique_x) - 1, MAX_TICKS, dtype=int)
            ticks = [unique_x[i] for i in idx]

        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2g'))


    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.06 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    handles, labels = ax.get_legend_handles_labels()
    n_items = len(labels)
    ncol = min(max(n_items, 1), 3)
    fig.legend(
        handles,
        labels,
        ncol=ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.6,
    )

    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)

    return out_base + ".pdf", out_base + ".png"


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build CSV from PKLs, and/or plot from CSV (publication-quality).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build-csv
    p1 = sub.add_parser("build-csv", help="Scan exp_*.pkl and write a summary CSV.")
    p1.add_argument("--dir", required=True, help="Directory containing exp_*.pkl")
    p1.add_argument("--out_csv", default="figures/curves_summary.csv", help="Output CSV path")
    p1.add_argument("--sigma_clip", action="store_true", help="Apply sigma clipping (default 3-sigma)")
    p1.add_argument("--sigma", type=float, default=3.0, help="Sigma threshold for sigma clipping")
    p1.add_argument("--extreme_method", choices=["trim", "winsorize", "none"], default="trim",
                    help="How to handle extremes when enabled")
    p1.add_argument("--trim_k", type=int, default=3, help="k for trim/winsorize (drop/cap k min and k max)")

    # plot-csv
    p2 = sub.add_parser("plot-csv", help="Plot curves directly from a CSV (no pkl needed).")
    p2.add_argument("--csv", required=True, help="Input CSV path (from build-csv or your own)")
    p2.add_argument("--out_fig", default="figures/curves.png", help="Output figure path (stem used; saves .pdf and .png)")
    p2.add_argument("--show_band", action="store_true", help="Shade error band using err_pct column")

    args = parser.parse_args()

    if args.cmd == "build-csv":
        df, comps, params, nfiles = build_csv_from_dir(
            in_dir=args.dir,
            out_csv=args.out_csv,
            sigma_clip=args.sigma_clip,
            sigma=args.sigma,
            extreme_method=args.extreme_method,
            trim_k=args.trim_k,
        )
        print(f"[OK] Processed {nfiles} files")
        print(f"[OK] Param names seen: {params}")
        print(f"[OK] Comparators seen: {comps}")
        print(f"[OK] Saved CSV -> {args.out_csv}")

    elif args.cmd == "plot-csv":
        pdf_path, png_path = plot_from_csv(
            csv_path=args.csv,
            out_fig=args.out_fig,
            show_band=args.show_band,
        )
        print(f"[OK] Read CSV  -> {args.csv}")
        print(f"[OK] Saved plot -> {pdf_path}")
        print(f"[OK] Saved plot -> {png_path}")


if __name__ == "__main__":
    main()
