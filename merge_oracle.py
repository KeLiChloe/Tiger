"""
Merge oracle pkl files into algo pkl files.

For each exp_d_XX.pkl + exp_d_XX_oracle.pkl pair:
  1. Verify seed sequences match (sanity check).
  2. Add oracle_profits_impl list from oracle pkl into algo pkl.
  3. Save as exp_d_XX_with_oracle.pkl.
"""

import pickle
import os
import glob
import sys


def merge_pair(algo_path: str, oracle_path: str, out_path: str) -> None:
    with open(algo_path, "rb") as f:
        algo_data = pickle.load(f)
    with open(oracle_path, "rb") as f:
        oracle_data = pickle.load(f)

    algo_seeds   = algo_data["seed"]
    oracle_seeds = oracle_data["seed"]

    if len(algo_seeds) != len(oracle_seeds):
        raise ValueError(
            f"{os.path.basename(algo_path)}: seed list length mismatch "
            f"({len(algo_seeds)} vs {len(oracle_seeds)})"
        )
    if algo_seeds != oracle_seeds:
        # Seeds differ — try to match by value and reorder oracle profits
        seed_to_oracle = {s: v for s, v in zip(oracle_seeds, oracle_data["oracle_profits_impl"])}
        try:
            oracle_profits_aligned = [seed_to_oracle[s] for s in algo_seeds]
        except KeyError as e:
            raise ValueError(
                f"{os.path.basename(algo_path)}: seed {e} in algo file "
                f"not found in oracle file; cannot align."
            )
        print(f"  ⚠️  Seeds differ — aligned by value.")
    else:
        oracle_profits_aligned = oracle_data["oracle_profits_impl"]

    merged = dict(algo_data)
    merged["oracle_profits_impl"] = oracle_profits_aligned

    with open(out_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"  ✅  {os.path.basename(out_path)}  "
          f"(N_sims={len(algo_seeds)}, "
          f"oracle mean={sum(oracle_profits_aligned)/len(oracle_profits_aligned):.2f})")


def main(directory: str) -> None:
    algo_files = sorted(glob.glob(os.path.join(directory, "exp_d_[0-9][0-9].pkl")))

    if not algo_files:
        print(f"No matching exp_d_XX.pkl files found in {directory}")
        return

    for algo_path in algo_files:
        base = os.path.basename(algo_path)          # exp_d_01.pkl
        tag  = base.replace(".pkl", "")             # exp_d_01
        oracle_path = os.path.join(directory, f"{tag}_oracle.pkl")
        out_path    = os.path.join(directory, f"{tag}_with_oracle.pkl")

        if not os.path.exists(oracle_path):
            print(f"  ⚠️  Skipping {base}: no matching oracle file found.")
            continue

        print(f"Merging {base} + {tag}_oracle.pkl ...")
        try:
            merge_pair(algo_path, oracle_path, out_path)
        except Exception as e:
            print(f"  ❌  Error: {e}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else \
        "exp_feb_2026/discrete/varying_d_set8"
    print(f"Directory: {directory}\n")
    main(directory)
    print("\nDone.")
