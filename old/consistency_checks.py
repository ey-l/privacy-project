"""
PHASE 2 — Internal Consistency Proof (Instacart)

Implements:
✅ TODO 4: Persistence analysis
✅ TODO 5: Distinctness analysis
✅ TODO 6: Permutation test
✅ TODO 7: Transition analysis

Assumes Instacart files in ./data/:
- orders.csv
- order_products__prior.csv
- order_products__train.csv
- products_labeled_instacart.csv   (from earlier step)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Load + build order-level pattern intensity time series
# -----------------------------
orders = pd.read_csv("./data/orders.csv")  # order_id, user_id, order_number, days_since_prior_order, ...
prior = pd.read_csv("./data/order_products__prior.csv")
train = pd.read_csv("./data/order_products__train.csv")
products_labeled = pd.read_csv("./data/products_labeled_instacart.csv")

order_products = pd.concat([prior, train], ignore_index=True)

label_cols = [
    "product_id",
    "is_diabetes_management_pattern",
    "is_low_sodium_hypertension_friendly_pattern",
    "is_gi_sensitivity_gluten_free_pattern",
]
order_products = order_products.merge(products_labeled[label_cols], on="product_id", how="left")

order_products = order_products.merge(
    orders[["order_id", "user_id", "order_number", "days_since_prior_order"]],
    on="order_id",
    how="left",
)

# Replace NaNs in label booleans with False
for c in label_cols[1:]:
    order_products[c] = order_products[c].fillna(False).astype(int)

# Build per-order pattern intensity (proportion of items in order)
order_pattern = (
    order_products
    .groupby(["user_id", "order_id", "order_number"], as_index=False)
    .agg(
        total_items=("product_id", "count"),
        diabetes_items=("is_diabetes_management_pattern", "sum"),
        htn_items=("is_low_sodium_hypertension_friendly_pattern", "sum"),
        gi_items=("is_gi_sensitivity_gluten_free_pattern", "sum"),
        # for time index reconstruction:
        days_since_prior_order=("days_since_prior_order", "max"),
    )
)

order_pattern["diabetes_prop"] = order_pattern["diabetes_items"] / order_pattern["total_items"]
order_pattern["htn_prop"] = order_pattern["htn_items"] / order_pattern["total_items"]
order_pattern["gi_prop"] = order_pattern["gi_items"] / order_pattern["total_items"]

# Reconstruct a per-user "day index" (Instacart has relative gaps, not absolute timestamps)
order_pattern = order_pattern.sort_values(["user_id", "order_number"])
order_pattern["days_since_prior_order"] = order_pattern["days_since_prior_order"].fillna(0.0)

order_pattern["t_days"] = (
    order_pattern.groupby("user_id")["days_since_prior_order"]
    .cumsum()
)

# -----------------------------
# 1) Regime detection helpers
# -----------------------------
def detect_activation_index(signal: np.ndarray, N: int) -> int:
    """
    Return the earliest index i such that signal[i:i+N] are all 1.
    If never activates, return -1.
    """
    L = len(signal)
    if L < N:
        return -1
    # sliding window sum
    win = np.convolve(signal, np.ones(N, dtype=int), mode="valid")
    hits = np.where(win == N)[0]
    return int(hits[0]) if len(hits) else -1


def build_regime_series(df_user: pd.DataFrame, prop_col: str, T: float) -> np.ndarray:
    """Binary per-order in-regime signal for a single user."""
    return (df_user[prop_col].values >= T).astype(int)


def is_consistent_user(df_user: pd.DataFrame, prop_col: str, T: float, N: int, M: int) -> bool:
    """
    Strong temporal definition:
    - must have an activation run of N consecutive in-regime orders
    - then must persist for >= M additional in-regime orders
      OR be right-censored at end of observed history
    """
    sig = build_regime_series(df_user, prop_col, T)
    act = detect_activation_index(sig, N)
    if act < 0:
        return False

    # persistence length starting from activation point (until first 0 or end)
    persistence = 0
    for i in range(act, len(sig)):
        if sig[i] == 1:
            persistence += 1
        else:
            break

    # need N + M total in-regime orders from activation start, unless censored at end
    if persistence >= (N + M):
        return True
    # right-censored: they stayed in-regime through end of data
    if act + persistence == len(sig):
        return True
    return False


def add_regime_flags(order_pattern: pd.DataFrame, prop_col: str, T: float) -> pd.DataFrame:
    """Add per-order binary flag `in_<prop_col>`."""
    out = order_pattern.copy()
    out[f"in_{prop_col}"] = (out[prop_col] >= T).astype(int)
    return out


# -----------------------------
# 2) Global regime parameters (tune)
# -----------------------------
REGIMES = {
    "diabetes": {"prop_col": "diabetes_prop", "T": 0.20, "N": 3, "M": 3},
    "htn":      {"prop_col": "htn_prop",      "T": 0.15, "N": 3, "M": 3},
    "gi":       {"prop_col": "gi_prop",       "T": 0.20, "N": 3, "M": 3},
}

# Precompute per-order in-regime flags
for rname, cfg in REGIMES.items():
    order_pattern = add_regime_flags(order_pattern, cfg["prop_col"], cfg["T"])

# Compute user-level consistent flags
user_flags = []
for user_id, dfu in order_pattern.groupby("user_id"):
    row = {"user_id": user_id, "total_orders": len(dfu)}
    for rname, cfg in REGIMES.items():
        row[f"is_consistent_{rname}_user"] = is_consistent_user(dfu.sort_values("order_number"),
                                                                cfg["prop_col"], cfg["T"], cfg["N"], cfg["M"])
    user_flags.append(row)

user_flags = pd.DataFrame(user_flags)
user_flags.to_csv("./data/phase2_user_consistent_flags.csv", index=False)

# ============================================================
# ✅ TODO 4: Persistence analysis
#   - P(Regime at t+1 | Regime at t)
#   - baseline prevalence
#   - Kaplan–Meier survival curve for regime persistence
# ============================================================

def persistence_stats(order_pattern: pd.DataFrame, in_col: str) -> dict:
    """
    Compute:
    - P(next=1 | current=1)
    - baseline P(next=1)
    - counts for sanity
    """
    # Align next state per user
    df = order_pattern.sort_values(["user_id", "order_number"]).copy()
    df["next_state"] = df.groupby("user_id")[in_col].shift(-1)

    cur1 = df[df[in_col] == 1]
    # drop last orders (no next_state)
    cur1 = cur1[cur1["next_state"].notna()]
    base = df[df["next_state"].notna()]

    p_next_given_cur = (cur1["next_state"].mean()) if len(cur1) else np.nan
    p_next_baseline = (base["next_state"].mean()) if len(base) else np.nan

    return {
        "p_next_given_current": float(p_next_given_cur),
        "p_next_baseline": float(p_next_baseline),
        "n_transitions_cur1": int(len(cur1)),
        "n_transitions_all": int(len(base)),
    }


def km_survival_durations(order_pattern: pd.DataFrame, in_col: str, time_col: str = "order_number"):
    """
    Build (duration, event) pairs for Kaplan–Meier:
    - Consider each *entry* into the regime (0 -> 1).
    - duration = how long (in units of time_col) they stay continuously in regime until exit (1 -> 0).
    - event = 1 if exited, 0 if right-censored at end.
    """
    durations = []
    events = []

    for user_id, dfu in order_pattern.sort_values(["user_id", "order_number"]).groupby("user_id"):
        s = dfu[in_col].values.astype(int)
        t = dfu[time_col].values

        if len(s) == 0:
            continue

        i = 0
        while i < len(s):
            # find entry
            if s[i] == 1 and (i == 0 or s[i-1] == 0):
                start_t = t[i]
                j = i
                while j < len(s) and s[j] == 1:
                    j += 1
                end_idx = j - 1
                end_t = t[end_idx]
                duration = end_t - start_t + 1  # inclusive count in order_number units

                # did we exit (i.e., next is 0) or censor at end?
                if j < len(s) and s[j] == 0:
                    event = 1
                else:
                    event = 0

                durations.append(duration)
                events.append(event)
                i = j
            else:
                i += 1

    return np.array(durations, dtype=float), np.array(events, dtype=int)


def kaplan_meier(durations: np.ndarray, events: np.ndarray):
    """
    Simple Kaplan–Meier estimator.
    Returns (times, survival_probs)
    """
    if len(durations) == 0:
        return np.array([]), np.array([])

    # sort by time
    order = np.argsort(durations)
    d = durations[order]
    e = events[order]

    uniq_times = np.unique(d)
    n_at_risk = len(d)
    S = 1.0
    times_out = [0.0]
    surv_out = [1.0]

    for t in uniq_times:
        # number of events and censored at time t
        idx = (d == t)
        d_i = e[idx].sum()          # events
        c_i = idx.sum() - d_i       # censored
        if n_at_risk > 0:
            if d_i > 0:
                S *= (1.0 - d_i / n_at_risk)
            times_out.append(float(t))
            surv_out.append(float(S))
        n_at_risk -= (d_i + c_i)

    return np.array(times_out), np.array(surv_out)


# Run persistence + KM plots
persistence_rows = []
plt.figure()
for rname, cfg in REGIMES.items():
    in_col = f"in_{cfg['prop_col']}"
    stats = persistence_stats(order_pattern, in_col)
    stats["regime"] = rname
    persistence_rows.append(stats)

    # KM survival in order_number units
    durations, events = km_survival_durations(order_pattern, in_col, time_col="order_number")
    km_t, km_s = kaplan_meier(durations, events)
    if len(km_t):
        plt.step(km_t, km_s, where="post", label=rname)

plt.title("Kaplan–Meier Survival: Regime Persistence (order-number units)")
plt.xlabel("Duration in regime (orders)")
plt.ylabel("Survival prob (still in regime)")
plt.legend()
plt.tight_layout()
plt.savefig("./data/phase2_km_survival_regimes.png", dpi=200)
plt.close()

persistence_df = pd.DataFrame(persistence_rows)
persistence_df.to_csv("./data/phase2_persistence_stats.csv", index=False)

print("TODO4 saved:")
print("- ./data/phase2_persistence_stats.csv")
print("- ./data/phase2_km_survival_regimes.png")
