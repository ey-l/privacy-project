#!/usr/bin/env python3
"""
identify_users.py

Pipeline:
  1. Load per-order proportions (output of identify_orders.py).
  2. Compute the p95 threshold for each prop_* column across all orders.
  3. Flag each order where prop > p95  →  order_flagged_{type}.
  4. Per user: compute the fraction of their orders that are flagged for each type
     →  user_prop_{type}.
  5. Compute user-level percentiles (p50/p75/p90/p95) of user_prop_{type}.
  6. Flag each user where user_prop > each user-level percentile threshold.
  7. Save results and print summary tables.

Outputs (all relative to --data_dir):
  processed/user_type_proportions.csv   wide table: one row per user
  processed/user_flags.csv              long table: one row per (user, type, threshold)
  processed/persistence_stats.csv       per-type transition probabilities and lift
"""

import argparse
from pathlib import Path

import pandas as pd

FLAG_COLS = [
    "is_diabetes_management",
    "is_low_sodium_hypertension_friendly",
    "is_gi_sensitivity_gluten_free",
    "is_lactose_free_or_lactase",
    "is_alcohol",
    "is_ultra_processed_or_sugary_beverage_heavy",
]

PERCENTILE_LEVELS = [0.50, 0.75, 0.90, 0.95]
P_LABELS = ["p50", "p75", "p90", "p95"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify users by purchasing pattern.")
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument(
        "--orders_csv",
        default="processed/order_type_proportions.csv",
        help="Per-order proportions CSV (relative to data_dir)",
    )
    parser.add_argument(
        "--out_users",
        default="processed/user_type_proportions.csv",
        help="Output: per-user wide table (relative to data_dir)",
    )
    parser.add_argument(
        "--out_flags",
        default="processed/user_flags.csv",
        help="Output: long-form user flags (relative to data_dir)",
    )
    parser.add_argument(
        "--out_persistence",
        default="processed/persistence_stats.csv",
        help="Output: persistence/transition stats (relative to data_dir)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    orders_path = data_dir / args.orders_csv
    out_users_path = data_dir / args.out_users
    out_flags_path = data_dir / args.out_flags
    out_persistence_path = data_dir / args.out_persistence

    prop_cols = [c.replace("is_", "prop_") for c in FLAG_COLS]
    type_names = [c.replace("is_", "") for c in FLAG_COLS]

    # ── 1. Load per-order proportions ─────────────────────────────────────────
    print(f"Loading {orders_path} ...")
    orders = pd.read_csv(orders_path)

    missing = [c for c in prop_cols + ["user_id"] if c not in orders.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.orders_csv}: {missing}")

    # ── 2. Order-level p95 thresholds ─────────────────────────────────────────
    order_p95 = orders[prop_cols].quantile(0.95)

    print("\nOrder-level p95 thresholds (used to flag individual orders):")
    print(f"  {'column':<52} {'p95':>7}")
    print("  " + "-" * 62)
    for pc in prop_cols:
        print(f"  {pc:<52} {order_p95[pc]:>7.4f}")

    # ── 3. Flag each order: prop > p95 ────────────────────────────────────────
    order_flag_cols = []
    for pc in prop_cols:
        t = pc.replace("prop_", "")
        fc = f"order_flagged_{t}"
        orders[fc] = orders[pc] > order_p95[pc]
        order_flag_cols.append(fc)

    # ── 4. Per-user fraction of flagged orders ────────────────────────────────
    user_stats = (
        orders.groupby("user_id")[order_flag_cols]
        .mean()  # mean of bool = fraction of True
        .reset_index()
    )
    rename_map = {f"order_flagged_{t}": f"user_prop_{t}" for t in type_names}
    user_stats = user_stats.rename(columns=rename_map)
    user_prop_cols = [f"user_prop_{t}" for t in type_names]

    # attach order count per user
    n_orders = orders.groupby("user_id").size().rename("n_orders").reset_index()
    user_stats = n_orders.merge(user_stats, on="user_id", how="left")

    # ── 5. User-level percentiles ─────────────────────────────────────────────
    user_pcts = user_stats[user_prop_cols].quantile(PERCENTILE_LEVELS)

    print("\nUser-level percentiles (fraction of flagged orders per user):")
    print(f"  {'column':<52} {'p50':>7} {'p75':>7} {'p90':>7} {'p95':>7}")
    print("  " + "-" * 84)
    for upc in user_prop_cols:
        vals = [user_pcts.loc[lv, upc] for lv in PERCENTILE_LEVELS]
        print(f"  {upc:<52} " + " ".join(f"{v:>7.4f}" for v in vals))

    # ── 6. Flag users at each percentile threshold ────────────────────────────
    for upc in user_prop_cols:
        t = upc.replace("user_prop_", "")
        for lv, lbl in zip(PERCENTILE_LEVELS, P_LABELS):
            user_stats[f"user_flag_{t}_{lbl}"] = user_stats[upc] > user_pcts.loc[lv, upc]

    # ── 7. Save wide table ────────────────────────────────────────────────────
    out_users_path.parent.mkdir(parents=True, exist_ok=True)
    user_stats.to_csv(out_users_path, index=False)
    print(f"\nWrote: {args.out_users} (rows={len(user_stats):,})")

    # ── 8. Print flagged-user counts ──────────────────────────────────────────
    print("\nFlagged user counts per type and threshold:")
    print(f"  {'type':<44} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8}")
    print("  " + "-" * 80)
    for t in type_names:
        counts = [int(user_stats[f"user_flag_{t}_{lbl}"].sum()) for lbl in P_LABELS]
        print(f"  {t:<44} " + " ".join(f"{c:>8,}" for c in counts))

    # ── 9. Save long-form flags table ─────────────────────────────────────────
    long_rows = []
    for t in type_names:
        upc = f"user_prop_{t}"
        for lv, lbl in zip(PERCENTILE_LEVELS, P_LABELS):
            flag_col = f"user_flag_{t}_{lbl}"
            tmp = user_stats.loc[user_stats[flag_col], ["user_id", "n_orders", upc]].copy()
            tmp["type"] = t
            tmp["threshold_level"] = lbl
            tmp["threshold_value"] = float(user_pcts.loc[lv, upc])
            tmp = tmp.rename(columns={upc: "user_prop"})
            long_rows.append(tmp)

    user_flags_long = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(
            columns=["user_id", "n_orders", "user_prop", "type", "threshold_level", "threshold_value"]
        )
    )
    user_flags_long.to_csv(out_flags_path, index=False)
    print(f"Wrote: {args.out_flags} (rows={len(user_flags_long):,})")

    # ── 10. Stats about user_flags_long ───────────────────────────────────────
    print(f"\nTotal unique users flagged (any type, any threshold): "
          f"{user_flags_long['user_id'].nunique():,}")

    print("\nFlagged users per type × threshold  (unique users | mean user_prop | max user_prop):")
    print(f"  {'type':<44} {'threshold':>9} {'n_users':>8} {'mean_prop':>10} {'max_prop':>9}")
    print("  " + "-" * 84)
    for t in type_names:
        for lbl in P_LABELS:
            sub = user_flags_long[
                (user_flags_long["type"] == t) & (user_flags_long["threshold_level"] == lbl)
            ]
            n = sub["user_id"].nunique()
            if n == 0:
                mean_p, max_p = 0.0, 0.0
            else:
                mean_p = sub["user_prop"].mean()
                max_p = sub["user_prop"].max()
            print(f"  {t:<44} {lbl:>9} {n:>8,} {mean_p:>10.4f} {max_p:>9.4f}")

    # ── 11. Persistence check: P(flagged t+1 | flagged t) ─────────────────────
    # Requires order_number to be present for within-user temporal ordering.
    if "order_number" not in orders.columns:
        print("\n[persistence] Skipped — 'order_number' column not found in orders CSV.")
    else:
        # Sort so consecutive rows within a user are chronological.
        orders_sorted = orders.sort_values(["user_id", "order_number"])

        persist_rows = []
        print("\nPersistence check: P(order flagged at t+1 | order flagged at t) vs. baseline")
        print(f"  {'type':<44} {'P(F|F)':>8} {'P(F|¬F)':>8} {'baseline':>9} {'lift':>6} {'n(F→F)':>9} {'n(F→¬F)':>8}")
        print("  " + "-" * 100)

        for t, fc in zip(type_names, order_flag_cols):
            grp = orders_sorted.groupby("user_id")[fc]
            # Shift within user: next order's flag
            shifted = orders_sorted.copy()
            shifted["_next"] = grp.shift(-1)  # NaN for last order of each user
            shifted = shifted.dropna(subset=["_next"])
            shifted["_next"] = shifted["_next"].astype(bool)

            cur = shifted[fc]
            nxt = shifted["_next"]

            n_ff = int((cur & nxt).sum())          # flagged → flagged
            n_fn = int((cur & ~nxt).sum())         # flagged → not-flagged
            n_nf = int((~cur & nxt).sum())         # not-flagged → flagged
            n_nn = int((~cur & ~nxt).sum())        # not-flagged → not-flagged

            n_f_total = n_ff + n_fn               # transitions starting from flagged
            n_nf_total = n_nf + n_nn              # transitions starting from not-flagged
            baseline = (n_ff + n_nf) / (n_ff + n_fn + n_nf + n_nn) if (n_ff + n_fn + n_nf + n_nn) > 0 else float("nan")

            p_flag_given_flag   = n_ff / n_f_total  if n_f_total  > 0 else float("nan")
            p_flag_given_noflag = n_nf / n_nf_total if n_nf_total > 0 else float("nan")
            lift = p_flag_given_flag / baseline if baseline > 0 else float("nan")

            print(f"  {t:<44} {p_flag_given_flag:>8.4f} {p_flag_given_noflag:>8.4f} {baseline:>9.4f} {lift:>6.2f} {n_ff:>9,} {n_fn:>8,}")

            persist_rows.append({
                "type": t,
                "p_flag_given_flag": p_flag_given_flag,
                "p_flag_given_noflag": p_flag_given_noflag,
                "baseline": baseline,
                "lift": lift,
                "n_ff": n_ff,
                "n_fn": n_fn,
                "n_nf": n_nf,
                "n_nn": n_nn,
            })

        persistence_df = pd.DataFrame(persist_rows)
        persistence_df.to_csv(out_persistence_path, index=False)
        print(f"\nWrote: {args.out_persistence} (rows={len(persistence_df):,})")

        print("\nFull persistence stats:")
        print(f"  {'type':<44} {'P(F|F)':>8} {'P(F|¬F)':>8} {'baseline':>9} {'lift':>6} "
              f"{'n_ff':>9} {'n_fn':>8} {'n_nf':>8} {'n_nn':>9}")
        print("  " + "-" * 115)
        for _, row in persistence_df.iterrows():
            print(
                f"  {row['type']:<44} {row['p_flag_given_flag']:>8.4f} "
                f"{row['p_flag_given_noflag']:>8.4f} {row['baseline']:>9.4f} "
                f"{row['lift']:>6.2f} {int(row['n_ff']):>9,} {int(row['n_fn']):>8,} "
                f"{int(row['n_nf']):>8,} {int(row['n_nn']):>9,}"
            )

        # ── 12. Exit-regime check: are F→¬F users just noisy or truly different? ──
        # For each type, split users into:
        #   "persistent"  — had at least one F→F transition  (never exited, or mostly stayed)
        #   "exiting"     — had at least one F→¬F transition but no F→F  (exited every time they entered)
        #   "both"        — had both F→F and F→¬F transitions
        # Then compare their overall user_prop distributions.
        print("\nExit-regime check: user_prop distribution by transition profile")
        print("(Are users who exit the regime actually lower-signal users overall?)")
        print(f"\n  {'type':<44} {'profile':<12} {'n_users':>8} {'mean_prop':>10} {'med_prop':>9} {'p75_prop':>9} {'p95_prop':>9}")
        print("  " + "-" * 105)

        for t, fc in zip(type_names, order_flag_cols):
            upc = f"user_prop_{t}"
            shifted = orders_sorted.copy()
            shifted["_next"] = orders_sorted.groupby("user_id")[fc].shift(-1)
            shifted = shifted.dropna(subset=["_next"])
            shifted["_next"] = shifted["_next"].astype(bool)
            cur = shifted[fc]
            nxt = shifted["_next"]

            users_ff  = set(shifted.loc[ cur &  nxt, "user_id"])
            users_fn  = set(shifted.loc[ cur & ~nxt, "user_id"])

            # Classify each user
            persistent = users_ff - users_fn          # only F→F
            exiting    = users_fn - users_ff          # only F→¬F (always exits)
            both       = users_ff & users_fn          # mixed

            for label, uid_set in [("persistent", persistent), ("exiting", exiting), ("both", both)]:
                if not uid_set:
                    continue
                props = user_stats.loc[user_stats["user_id"].isin(uid_set), upc]
                print(
                    f"  {t:<44} {label:<12} {len(props):>8,} "
                    f"{props.mean():>10.4f} {props.median():>9.4f} "
                    f"{props.quantile(0.75):>9.4f} {props.quantile(0.95):>9.4f}"
                )


if __name__ == "__main__":
    main()
