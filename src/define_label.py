"""
Retention Risk - Phase 2: Define the Inactivity Window & Label
Reads data/cleaned_transactions.parquet.

Approach 
1. Compute inter-purchase gaps across all customers to find a defensible window.
2. Set a snapshot date roughly mid-way through the dataset.
3. Label customers "at risk" if their gap since last purchase (as of snapshot)
   exceeds the chosen window.
4. Validate: check whether "at risk" customers actually stayed quiet during
   the holdout period (snapshot -> end of data). This tells us if the label
   is actually predictive, not just descriptive.
"""

import pandas as pd
import numpy as np

CLEANED_PATH = "data/cleaned_transactions.parquet"


def load_cleaned(path: str = CLEANED_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def inter_purchase_gaps(df: pd.DataFrame) -> pd.Series:
    
    purchase_days = (
        df.groupby("CustomerID")["InvoiceDate"]
        .apply(lambda x: pd.Series(sorted(x.dt.normalize().unique())))
    )

    gaps = []
    for _, dates in df.groupby("CustomerID")["InvoiceDate"]:
        days = sorted(dates.dt.normalize().unique())
        if len(days) > 1:
            diffs = np.diff(days) / np.timedelta64(1, "D")
            gaps.extend(diffs)

    return pd.Series(gaps, name="gap_days")


def summarise_gaps(gaps: pd.Series) -> None:
    print("=== Inter-purchase gap distribution (days) ===")
    print(gaps.describe(percentiles=[0.5, 0.75, 0.9, 0.95]))


def label_customers(df: pd.DataFrame, snapshot_date: pd.Timestamp,
                     window_days: int) -> pd.DataFrame:
    pre_snapshot = df[df["InvoiceDate"] <= snapshot_date]

    last_purchase = (
        pre_snapshot.groupby("CustomerID")["InvoiceDate"]
        .max()
        .reset_index()
        .rename(columns={"InvoiceDate": "LastPurchaseDate"})
    )

    last_purchase["RecencyDays"] = (
        snapshot_date - last_purchase["LastPurchaseDate"]
    ).dt.days

    last_purchase["AtRisk"] = (last_purchase["RecencyDays"] > window_days).astype(int)

    return last_purchase


def validate_against_holdout(labels: pd.DataFrame, df: pd.DataFrame,
                              snapshot_date: pd.Timestamp) -> None:
    
    holdout = df[df["InvoiceDate"] > snapshot_date]
    returned_customers = set(holdout["CustomerID"].unique())

    labels = labels.copy()
    labels["ReturnedInHoldout"] = labels["CustomerID"].isin(returned_customers)

    print("\n=== Holdout validation ===")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Holdout period: {snapshot_date.date()} onward\n")

    summary = labels.groupby("AtRisk")["ReturnedInHoldout"].agg(["mean", "count"])
    summary.columns = ["Pct_Returned_In_Holdout", "N_Customers"]
    print(summary)

    print(f"\nInterpretation: AtRisk=1 customers should have a LOW "
          f"return rate, AtRisk=0 should have a HIGH return rate. "
          f"A big gap between the two rows means the label is predictive.")


def sweep_windows(df: pd.DataFrame, snapshot_date: pd.Timestamp,
                   candidate_windows: list) -> pd.DataFrame:
    
    holdout = df[df["InvoiceDate"] > snapshot_date]
    returned_customers = set(holdout["CustomerID"].unique())

    rows = []
    for w in candidate_windows:
        labels = label_customers(df, snapshot_date, w)
        labels["ReturnedInHoldout"] = labels["CustomerID"].isin(returned_customers)

        pct_at_risk = labels["AtRisk"].mean()
        n_at_risk = labels["AtRisk"].sum()
        n_not_at_risk = len(labels) - n_at_risk

        return_rate_at_risk = labels.loc[labels["AtRisk"] == 1, "ReturnedInHoldout"].mean()
        return_rate_not_at_risk = labels.loc[labels["AtRisk"] == 0, "ReturnedInHoldout"].mean()

        actually_quiet = labels[~labels["ReturnedInHoldout"]]
        recall = (actually_quiet["AtRisk"] == 1).mean() if len(actually_quiet) else np.nan

        rows.append({
            "window_days": w,
            "pct_labeled_at_risk": round(pct_at_risk, 3),
            "n_at_risk": int(n_at_risk),
            "n_not_at_risk": int(n_not_at_risk),
            "false_alarm_rate": round(return_rate_at_risk, 3),  # lower = better
            "retention_rate_not_at_risk": round(return_rate_not_at_risk, 3),  # higher = better
            "recall_of_actual_churners": round(recall, 3),  # higher = better
            "separation": round(return_rate_not_at_risk - return_rate_at_risk, 3),  # higher = better
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_cleaned()

    gaps = inter_purchase_gaps(df)
    summarise_gaps(gaps)

    
    min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    snapshot_date = min_date + (max_date - min_date) * 0.65
    print(f"\nSnapshot date (65% through the timeline): {snapshot_date.date()}")
    print(f"Full range: {min_date.date()} -> {max_date.date()}")

    candidate_windows = [60, 90, 120, 150, 180, 210,
                         int(round(gaps.quantile(0.90) / 10) * 10),
                         int(round(gaps.quantile(0.95) / 10) * 10)]
    candidate_windows = sorted(set(candidate_windows))

    print(f"\n=== Comparing candidate windows: {candidate_windows} ===")
    comparison = sweep_windows(df, snapshot_date, candidate_windows)
    print(comparison.to_string(index=False))

    CHOSEN_WINDOW = 120  # chosen: best balance of recall (73%) vs false alarms (40%);
                          # past this point recall drops faster than precision improves

    labels = label_customers(df, snapshot_date, CHOSEN_WINDOW)
    print(f"\n=== Final labels using window = {CHOSEN_WINDOW} days ===")
    print(labels["AtRisk"].value_counts(normalize=True))
    validate_against_holdout(labels, df, snapshot_date)

    labels.to_parquet("data/customer_labels.parquet", index=False)
    print("\nSaved: data/customer_labels.parquet")