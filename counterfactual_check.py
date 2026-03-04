#!/usr/bin/env python3
"""
counterfactual_check.py

Counterfactual sanity check: do the purchasing regimes disappear under
random permutation?

Method
------
1. Load per-order proportions. Sample N users.
2. Flag orders at the p95 proportion threshold (computed on the full dataset
   for stable thresholds, applied to the sample).
3. Compute *real* persistence for the sample:
     P(F|F) = P(order flagged at t+1 | order flagged at t)
     lift   = P(F|F) / baseline
4. Run K permutation rounds. In each round, shuffle the flagged-order
   sequence *within each user* (preserving each user's overall flagging
   rate but destroying temporal structure). Recompute P(F|F) and lift.
5. Report: real vs. permuted mean ± std, and an empirical p-value
   (fraction of permutations where permuted lift >= real lift).

A genuine regime should have real lift >> permuted lift and p ≈ 0.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

FLAG_COLS = [
    "is_diabetes_management",
    "is_low_sodium_hypertension_friendly",
    "is_gi_sensitivity_gluten_free",
    "is_lactose_free_or_lactase",
    "is_alcohol",
    "is_ultra_processed_or_sugary_beverage_heavy",
]


def compute_persistence(df: pd.DataFrame, flag_col: str) -> dict:
    """
    Given a DataFrame with columns [user_id, order_number, <flag_col>] sorted
    by (user_id, order_number), compute transition counts and P(F|F).
    """
    shifted = df[["user_id", flag_col]].copy()
    shifted["_next"] = df.groupby("user_id")[flag_col].shift(-1)
    shifted = shifted.dropna(subset=["_next"])
    shifted["_next"] = shifted["_next"].astype(bool)

    cur = shifted[flag_col]
    nxt = shifted["_next"]

    n_ff = int((cur & nxt).sum())
    n_fn = int((cur & ~nxt).sum())
    n_nf = int((~cur & nxt).sum())
    n_nn = int((~cur & ~nxt).sum())

    n_f_total = n_ff + n_fn
    n_total   = n_ff + n_fn + n_nf + n_nn

    baseline         = (n_ff + n_nf) / n_total   if n_total   > 0 else float("nan")
    p_flag_given_flag = n_ff / n_f_total          if n_f_total > 0 else float("nan")
    lift             = p_flag_given_flag / baseline if baseline > 0 else float("nan")

    return {"p_ff": p_flag_given_flag, "baseline": baseline, "lift": lift,
            "n_ff": n_ff, "n_fn": n_fn, "n_transitions": n_f_total}


def permuted_persistence(df: pd.DataFrame, flag_col: str, rng: np.random.Generator) -> dict:
    """
    Shuffle the flag sequence within each user, then recompute persistence.
    This preserves each user's overall flagging rate but destroys temporal order.
    """
    perm = df[["user_id", "order_number", flag_col]].copy()
    perm[flag_col] = (
        perm.groupby("user_id")[flag_col]
        .transform(lambda s: rng.permutation(s.values))
    )
    return compute_persistence(perm, flag_col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual sanity check for purchasing regimes.")
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--orders_csv",  default="processed/order_type_proportions.csv")
    parser.add_argument("--n_users",     type=int,   default=5_000,
                        help="Number of users to sample (default: 5000)")
    parser.add_argument("--n_perms",     type=int,   default=100,
                        help="Number of permutation rounds (default: 100)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--min_orders",  type=int,   default=5,
                        help="Min orders per user to be included (default: 5)")
    parser.add_argument("--out",         default="processed/counterfactual_results.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    prop_cols  = [c.replace("is_", "prop_") for c in FLAG_COLS]
    type_names = [c.replace("is_", "")      for c in FLAG_COLS]

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    print(f"Loading {data_dir / args.orders_csv} ...")
    orders = pd.read_csv(data_dir / args.orders_csv)

    required = prop_cols + ["user_id", "order_number"]
    missing = [c for c in required if c not in orders.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop users with too few orders (not enough transitions to measure)
    order_counts = orders.groupby("user_id").size()
    eligible = order_counts[order_counts >= args.min_orders].index
    orders = orders[orders["user_id"].isin(eligible)]
    print(f"Eligible users (>= {args.min_orders} orders): {len(eligible):,}")

    # ── 2. Sample users ───────────────────────────────────────────────────────
    all_users = orders["user_id"].unique()
    n_sample  = min(args.n_users, len(all_users))
    sampled   = rng.choice(all_users, size=n_sample, replace=False)
    sample    = orders[orders["user_id"].isin(sampled)].sort_values(["user_id", "order_number"])
    print(f"Sampled {n_sample:,} users ({len(sample):,} orders)\n")

    # ── 3. Compute p95 thresholds on the full dataset ─────────────────────────
    order_p95 = orders[prop_cols].quantile(0.95)

    order_flag_cols = []
    for pc in prop_cols:
        t  = pc.replace("prop_", "")
        fc = f"order_flagged_{t}"
        sample = sample.copy()
        sample[fc] = sample[pc] > order_p95[pc]
        order_flag_cols.append(fc)

    # ── 4. Real persistence on the sample ────────────────────────────────────
    real_results = {}
    for t, fc in zip(type_names, order_flag_cols):
        real_results[t] = compute_persistence(sample, fc)

    # ── 5. Permutation rounds ─────────────────────────────────────────────────
    print(f"Running {args.n_perms} permutation rounds ...")
    t0 = time.time()

    perm_lifts   = {t: [] for t in type_names}
    perm_pffs    = {t: [] for t in type_names}

    for i in range(args.n_perms):
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Round {i+1}/{args.n_perms}  ({elapsed:.1f}s elapsed)")
        for t, fc in zip(type_names, order_flag_cols):
            res = permuted_persistence(sample, fc, rng)
            perm_lifts[t].append(res["lift"])
            perm_pffs[t].append(res["p_ff"])

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"\nSample: {n_sample:,} users | {args.n_perms} permutations | "
          f"min_orders={args.min_orders} | p95 threshold | seed={args.seed}")

    header = (f"\n  {'type':<44} {'real_P(F|F)':>12} {'perm_P(F|F)':>12} "
              f"{'real_lift':>10} {'perm_lift':>10} {'perm_std':>9} {'p-value':>8}")
    print(header)
    print("  " + "-" * 115)

    rows = []
    for t in type_names:
        r  = real_results[t]
        pl = np.array(perm_lifts[t])
        pp = np.array(perm_pffs[t])
        pval = float((pl >= r["lift"]).mean())

        print(
            f"  {t:<44} {r['p_ff']:>12.4f} {pp.mean():>12.4f} "
            f"{r['lift']:>10.2f} {pl.mean():>10.2f} {pl.std():>9.3f} {pval:>8.3f}"
        )
        rows.append({
            "type":              t,
            "real_p_ff":         r["p_ff"],
            "real_lift":         r["lift"],
            "real_baseline":     r["baseline"],
            "real_n_ff":         r["n_ff"],
            "real_n_fn":         r["n_fn"],
            "perm_p_ff_mean":    pp.mean(),
            "perm_p_ff_std":     pp.std(),
            "perm_lift_mean":    pl.mean(),
            "perm_lift_std":     pl.std(),
            "p_value":           pval,
            "n_users_sampled":   n_sample,
            "n_perms":           args.n_perms,
        })

    out_path = data_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote: {args.out}")
    print("\nInterpretation: p-value = fraction of permutations where permuted lift >= real lift.")
    print("A genuine regime should have p ≈ 0.000 and real_lift >> perm_lift.")


if __name__ == "__main__":
    main()
