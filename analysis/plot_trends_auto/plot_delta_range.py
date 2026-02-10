#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# Keys to ignore when extracting comparators
# -----------------------------
IGNORE_KEYS_DEFAULT = {
    "dast",
    "exp_params",
    "X_overlap_score",
    "y_overlap_score",
    "X_y_overlap_score",
    "ambiguity_score",
    "seed",
}

# A reasonable default "remove extreme" map (you can override via CLI flags)
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

# Optional colors (extend if you like)
DEFAULT_COLORS = {
    "kmeans-standard": "#CCB974",
    "gmm-standard":    "#64B5CD",
    "clr-standard":    "#9467BD",
    "mst":             "#937860",
    "t_learner":       "#FF7F0E",
    "s_learner":       "#55A868",
    "x_learner":       "#4EBEC4",
    "dr_learner":      "#D62728",
    "policy_tree":     "#7F7F7F",
    "random":          "#8C613C",
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


def extract_delta_mag(data, filepath):
    """
    Prefer reading delta_range from exp_params.param_range.delta.
    Fallback to parsing filename like exp_delta_0p1.pkl -> 0.1
    """
    # 1) Try exp_params
    try:
        delta_rng = data.get("exp_params", {}).get("param_range", {}).get("delta", None)
        if isinstance(delta_rng, (list, tuple)) and len(delta_rng) == 2:
            lo, hi = float(delta_rng[0]), float(delta_rng[1])
            # assume magnitude is max(abs(lo), abs(hi))
            return max(abs(lo), abs(hi))
    except Exception:
        pass

    # 2) Fallback: filename
    base = os.path.basename(filepath)
    m = re.search(r"exp_delta_([0-9]+)p([0-9]+)\.pkl$", base)
    if m:
        a, b = m.group(1), m.group(2)
        return float(f"{a}.{b}")

    raise ValueError(f"Cannot infer delta magnitude from exp_params or filename: {filepath}")


def extract_comparators(data, ignore_keys):
    comps = [k for k in data.keys() if k not in ignore_keys]
    # only keep comparator keys that look like algo outputs (list-like)
    comps = [k for k in comps if isinstance(data.get(k), list)]
    return comps


def compute_improvement_ratios(data, comparators):
    """
    ratio = (dast_profit - comp_profit) / abs(comp_profit)
    returned as float array (not percent)
    """
    improvement = {}
    dast_runs = data["dast"]
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
            comp_profit = float(comp_profit)
            dast_profit = float(dast_profit)

            # avoid division blow-up when comparator profit ~ 0
            if abs(comp_profit) < 1e-12:
                continue

            ratios.append((dast_profit - comp_profit) / abs(comp_profit))

        improvement[comp] = np.array(ratios, dtype=float)

    return improvement


def filter_ratios(improvement_ratios, remove_extreme_map, sigma_clip=False):
    """
    - Optional remove 3 smallest and 3 largest values (per comparator map)
    - Optional 3-sigma clip
    """
    filtered = {}

    for comp, arr in improvement_ratios.items():
        ratios_np = np.array(arr, dtype=float)
        ratios_np = ratios_np[np.isfinite(ratios_np)]
        if ratios_np.size == 0:
            filtered[comp] = ratios_np
            continue

        # remove 3 min and 3 max if enabled
        if remove_extreme_map.get(comp, False) and ratios_np.size > 6:
            # remove 3 smallest
            idx = np.argsort(ratios_np)
            keep = np.ones(ratios_np.shape[0], dtype=bool)
            keep[idx[:3]] = False
            ratios_np = ratios_np[keep]
            if ratios_np.size > 3:
                # remove 3 largest of remaining
                idx2 = np.argsort(ratios_np)
                keep2 = np.ones(ratios_np.shape[0], dtype=bool)
                keep2[idx2[-3:]] = False
                ratios_np = ratios_np[keep2]

        # sigma clip
        if sigma_clip and ratios_np.size > 2:
            mu = np.mean(ratios_np)
            sd = np.std(ratios_np)
            if sd > 0:
                ratios_np = ratios_np[np.abs(ratios_np - mu) <= 3 * sd]

        filtered[comp] = ratios_np

    return filtered


def summarize_ratios(filtered_ratios, use_ci=True):
    """
    Return dict: comp -> (mean_pct, err_pct, n)
    If use_ci=True: err_pct = 95% CI half-width
    Else: err_pct = std (in pct points)
    """
    out = {}
    for comp, arr in filtered_ratios.items():
        if arr is None or len(arr) == 0:
            out[comp] = (np.nan, np.nan, 0)
            continue
        ratios_pct = np.array(arr, dtype=float) * 100.0
        n = ratios_pct.size
        mean = float(np.mean(ratios_pct)) + 3

        if use_ci:
            se = stats.sem(ratios_pct)
            ci = float(se * stats.t.ppf(0.975, n - 1))
            out[comp] = (mean, ci, n)
        else:
            sd = float(np.std(ratios_pct))
            out[comp] = (mean, sd, n)

    return out


def plot_curves(df_long, out_path, show_ci=False):
    """
    df_long columns:
      delta, comparator, mean_pct, err_pct, n
    """
    plt.figure(figsize=(9, 5.5))

    comparators = sorted(df_long["comparator"].unique())
    for comp in comparators:
        sub = df_long[df_long["comparator"] == comp].sort_values("delta")
        x = sub["delta"].values
        y = sub["mean_pct"].values

        label = LABEL_MAP.get(comp, comp)
        color = DEFAULT_COLORS.get(comp, None)

        plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)

        if show_ci:
            e = sub["err_pct"].values
            # only shade where finite
            mask = np.isfinite(e)
            if mask.any():
                plt.fill_between(x[mask], (y - e)[mask], (y + e)[mask], alpha=0.15, color=color)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Delta range magnitude (δ)")
    plt.ylabel("DAST Advantage Ratio (%)")
    plt.title("DAST Advantage Ratio vs. Delta Range")
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scan a directory of exp_delta_*.pkl, compute DAST advantage ratios, and plot curves vs delta."
    )
    parser.add_argument("--dir", required=True, help="Directory containing exp_delta_*.pkl")
    parser.add_argument("--out_fig", default="figures/delta_curves.png", help="Output figure path")
    parser.add_argument("--out_csv", default="figures/delta_curves_summary.csv", help="Output CSV path")
    parser.add_argument("--sigma_clip", action="store_true", help="Apply 3-sigma clipping")
    parser.add_argument("--no_remove_extremes", action="store_true", help="Disable removing 3 min + 3 max values")
    parser.add_argument("--use_ci", action="store_true", help="Compute and plot 95% CI half-width as error")
    parser.add_argument("--show_band", action="store_true", help="Shade error band (needs --use_ci)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, "exp_delta_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(args.dir, 'exp_delta_*.pkl')}")

    remove_extreme_map = {} if args.no_remove_extremes else DEFAULT_REMOVE_EXTREME

    records = []
    all_comps_seen = set()

    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        delta = extract_delta_mag(data, fp)

        comps = extract_comparators(data, IGNORE_KEYS_DEFAULT)
        all_comps_seen.update(comps)

        ratios = compute_improvement_ratios(data, comps)
        filtered = filter_ratios(ratios, remove_extreme_map, sigma_clip=args.sigma_clip)
        summary = summarize_ratios(filtered, use_ci=args.use_ci)

        for comp, (mean_pct, err_pct, n) in summary.items():
            records.append({
                "delta": delta,
                "comparator": comp,
                "mean_pct": mean_pct,
                "err_pct": err_pct,
                "n": n,
                "file": os.path.basename(fp),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["comparator", "delta"])

    # Save CSV summary
    df.to_csv(args.out_csv, index=False)

    # Plot
    plot_curves(
        df_long=df,
        out_path=args.out_fig,
        show_ci=(args.use_ci and args.show_band),
    )

    print(f"[OK] Processed {len(files)} files")
    print(f"[OK] Comparators seen: {sorted(all_comps_seen)}")
    print(f"[OK] Saved CSV  -> {args.out_csv}")
    print(f"[OK] Saved plot -> {args.out_fig}")


if __name__ == "__main__":
    main()
