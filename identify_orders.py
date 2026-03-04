"""
Now we need to identify the orders that have higher-than-median proportions of the respective product types. 
This is an intermediate step to identify the users that have the respective purchasing patterns. 
To identify the users, we will also use the temporal information across orders. 
But, for your current step, let's only identify the orders. 
Your task is to give me the Python script to identify the flagged orders 
"""

#!/usr/bin/env python3
"""
Identify "flagged orders" whose proportion of items in a given product-type
is higher than the *median* proportion across all orders (for that type).

Inputs:
  - product_type_flags.csv  (from previous step)
      columns:
        product_id,
        is_diabetes_management,
        is_low_sodium_hypertension_friendly,
        is_gi_sensitivity_gluten_free,
        is_lactose_free_or_lactase,
        is_alcohol,
        is_ultra_processed_or_sugary_beverage_heavy
  - order_products__prior.csv / order_products__train.csv
      columns (typical):
        order_id, product_id, add_to_cart_order, reordered
  - orders.csv
      columns (typical):
        order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order

Outputs:
  - order_type_proportions.csv: per-order proportions + "flagged" booleans
  - flagged_orders_long.csv: one row per (order_id, type) where flagged=True (useful for later temporal work)

Definition:
  For each order o and type t:
    prop(o,t) = (# line items in o flagged as type t) / (total # line items in o)
  flagged(o,t) = prop(o,t) > median_over_orders(prop(*,t))

Notes:
- Proportions are computed over line items (not unique products). If you want unique products per order,
  replace the denominator with nunique(product_id) and numerator accordingly.
- Median is computed over orders that exist in your joined order_products data.
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


def load_order_products(data_dir: Path) -> pd.DataFrame:
    parts = []
    for fn in ["order_products__prior.csv", "order_products__train.csv"]:
        p = data_dir / "raw" / fn
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(
            f"Could not find order_products__prior.csv or order_products__train.csv in {data_dir}"
        )
    op = pd.concat(parts, ignore_index=True)
    # keep only needed columns
    need = ["order_id", "product_id"]
    for c in need:
        if c not in op.columns:
            raise ValueError(f"{c} missing from order_products file(s)")
    return op[need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data/", help="Directory containing Instacart CSVs")
    ap.add_argument("--flags_csv", type=str, default="processed/products_flagged.csv", help="Product flags CSV")
    ap.add_argument("--out_orders", type=str, default="processed/order_type_proportions.csv")
    ap.add_argument("--out_flagged_long", type=str, default="processed/flagged_orders_long.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_orders_path = data_dir / args.out_orders
    out_flagged_long_path = data_dir / args.out_flagged_long

    prop_cols = [c.replace("is_", "prop_") for c in FLAG_COLS]

    if out_orders_path.exists() and (data_dir / args.flags_csv).exists():
        print(f"Cache found — loading {out_orders_path} directly (skipping recompute).")
        order_stats = pd.read_csv(out_orders_path)
    else:
        flags = pd.read_csv(data_dir / args.flags_csv)
        if "product_id" not in flags.columns:
            raise ValueError("products_flagged.csv must include product_id")
        missing = [c for c in FLAG_COLS if c not in flags.columns]
        if missing:
            raise ValueError(f"Missing flag columns in {args.flags_csv}: {missing}")

        op = load_order_products(data_dir)

        # Join line items to flags
        df = op.merge(flags[["product_id"] + FLAG_COLS], on="product_id", how="left")
        # Products not in flags get treated as False
        for c in FLAG_COLS:
            df[c] = df[c].fillna(False).astype(bool)

        # Per-order denominator: number of line items
        order_size = df.groupby("order_id").size().rename("n_items").reset_index()

        # Per-order numerator: number of flagged line items
        counts = df.groupby("order_id")[FLAG_COLS].sum().reset_index()  # bool sum -> counts

        order_stats = order_size.merge(counts, on="order_id", how="left")
        for c in FLAG_COLS:
            order_stats[c] = order_stats[c].fillna(0).astype(int)

        # Compute proportions
        for c in FLAG_COLS:
            prop_col = c.replace("is_", "prop_")
            order_stats[prop_col] = order_stats[c] / order_stats["n_items"]

        # Compute medians across orders for each proportion
        medians = order_stats[prop_cols].median(numeric_only=True)

        # flagged orders: proportion > median
        for pc in prop_cols:
            sc = pc.replace("prop_", "flagged_")
            order_stats[sc] = order_stats[pc] > float(medians[pc])

        # Attach order metadata if available
        orders_path = data_dir / "raw/orders.csv"
        if orders_path.exists():
            orders = pd.read_csv(orders_path)
            if "order_id" in orders.columns:
                meta_cols = [
                    c for c in ["order_id", "user_id", "eval_set", "order_number", "order_dow",
                                "order_hour_of_day", "days_since_prior_order"]
                    if c in orders.columns
                ]
                order_stats = order_stats.merge(orders[meta_cols], on="order_id", how="left")

        # Save wide per-order table
        out_orders_path.parent.mkdir(parents=True, exist_ok=True)
        order_stats.to_csv(out_orders_path, index=False)

        # Save long table: (order_id, type, proportion, median, flagged)
        long_rows = []
        for c in FLAG_COLS:
            t = c.replace("is_", "")
            pc = c.replace("is_", "prop_")
            sc = pc.replace("prop_", "flagged_")
            tmp = order_stats[["order_id", pc, sc]].copy()
            tmp["type"] = t
            tmp["median_threshold"] = float(medians[pc])
            tmp = tmp.rename(columns={pc: "proportion", sc: "flagged"})
            tmp = tmp[tmp["flagged"]]
            long_rows.append(tmp)

        flagged_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
            columns=["order_id", "type", "proportion", "flagged", "median_threshold"]
        )
        flagged_long.to_csv(out_flagged_long_path, index=False)

        print(f"Wrote: {args.out_orders} (rows={len(order_stats):,})")
        print(f"Wrote: {args.out_flagged_long} (rows={len(flagged_long):,})")

    percentiles = order_stats[prop_cols].quantile([0.50, 0.75, 0.90, 0.95])
    print("\nProportion percentiles per order:")
    print(f"  {'column':<48} {'p50':>7} {'p75':>7} {'p90':>7} {'p95':>7}")
    print("  " + "-" * 76)
    for pc in prop_cols:
        p50 = percentiles.loc[0.50, pc]
        p75 = percentiles.loc[0.75, pc]
        p90 = percentiles.loc[0.90, pc]
        p95 = percentiles.loc[0.95, pc]
        print(f"  {pc:<48} {p50:>7.4f} {p75:>7.4f} {p90:>7.4f} {p95:>7.4f}")

"""
Proportion percentiles per order:
  column                                               p50     p75     p90     p95
  ----------------------------------------------------------------------------
  prop_diabetes_management                          0.0000  0.0000  0.0476  0.1000
  prop_low_sodium_hypertension_friendly             0.0000  0.0000  0.0312  0.0833
  prop_gi_sensitivity_gluten_free                   0.0000  0.0000  0.0385  0.1000
  prop_lactose_free_or_lactase                      0.0000  0.0000  0.0909  0.1538
  prop_alcohol                                      0.0000  0.0000  0.0000  0.0000
  prop_ultra_processed_or_sugary_beverage_heavy     0.0938  0.2273  0.4000  0.5000
"""

if __name__ == "__main__":
    main()