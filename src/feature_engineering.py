"""
Retention Risk - Phase 3: RFM & Feature Engineering
Reads data/cleaned_transactions.parquet and data/customer_labels.parquet.

CRITICAL RULE: every feature is computed using ONLY transactions on or
before the snapshot date (same cutoff used for the label in Phase 2).
Using anything after the snapshot would leak future information into
features the model is supposed to predict from - the model would be
learning from data it wouldn't have in a real deployment.
"""

import pandas as pd
import numpy as np

CLEANED_PATH = "data/cleaned_transactions.parquet"
LABELS_PATH = "data/customer_labels.parquet"


def load_data():
    df = pd.read_parquet(CLEANED_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    labels = pd.read_parquet(LABELS_PATH)
    return df, labels


def get_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    return min_date + (max_date - min_date) * 0.65


def build_rfm(pre_snapshot: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Core RFM: Recency, Frequency, Monetary."""
    grouped = pre_snapshot.groupby("CustomerID")

    rfm = grouped.agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),          # distinct orders
        Monetary=("LineTotal", "sum"),
        TotalItems=("Quantity", "sum"),
        AvgUnitPrice=("Price", "mean"),
    ).reset_index()

    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"]).dt.days
    rfm["Tenure"] = (snapshot_date - rfm["FirstPurchaseDate"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["AvgBasketSize"] = rfm["TotalItems"] / rfm["Frequency"]

    return rfm


def build_diversity_features(pre_snapshot: pd.DataFrame) -> pd.DataFrame:
    """How varied a customer's purchases are — proxy for engagement breadth."""
    diversity = pre_snapshot.groupby("CustomerID").agg(
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()
    return diversity


def build_trend_feature(pre_snapshot: pd.DataFrame,
                         snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Roughly, is the customer spending more or less than before?"""
    results = []
    for cust_id, group in pre_snapshot.groupby("CustomerID"):
        group = group.sort_values("InvoiceDate")
        dates = group["InvoiceDate"]

        if dates.nunique() < 2:
            results.append({"CustomerID": cust_id, "SpendTrend": 0.0})
            continue

        midpoint = dates.min() + (dates.max() - dates.min()) / 2
        early = group[group["InvoiceDate"] <= midpoint]["LineTotal"].sum()
        late = group[group["InvoiceDate"] > midpoint]["LineTotal"].sum()

        # normalised trend, bounded in [-1, 1]. Use abs(early)+abs(late) as
        # the denominator rather than (early+late) -- LineTotal can be
        # negative (cancelled invoices), so early and late can nearly
        # cancel each other out, making a raw-sum denominator blow up
        # toward zero and the ratio explode. abs() keeps the denominator
        # safely away from zero unless both periods are truly empty.
        denom = abs(early) + abs(late)
        trend = (late - early) / denom if denom > 0 else 0.0
        trend = float(np.clip(trend, -1.0, 1.0))
        results.append({"CustomerID": cust_id, "SpendTrend": trend})

    return pd.DataFrame(results)


def build_feature_table(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = get_snapshot_date(df)
    pre_snapshot = df[df["InvoiceDate"] <= snapshot_date]

    rfm = build_rfm(pre_snapshot, snapshot_date)
    diversity = build_diversity_features(pre_snapshot)
    trend = build_trend_feature(pre_snapshot, snapshot_date)

    features = rfm.merge(diversity, on="CustomerID", how="left")
    features = features.merge(trend, on="CustomerID", how="left")

    features = features.merge(
        labels[["CustomerID", "AtRisk"]], on="CustomerID", how="inner"
    )

    # drop helper date columns, keep numeric features + label
    features = features.drop(columns=["LastPurchaseDate", "FirstPurchaseDate"])

    return features


if __name__ == "__main__":
    df, labels = load_data()
    snapshot_date = get_snapshot_date(df)
    print(f"Snapshot date: {snapshot_date.date()} (all features computed on-or-before this date)")

    features = build_feature_table(df, labels)

    print(f"\nFeature table shape: {features.shape}")
    print(f"\nColumns: {list(features.columns)}")
    print("\n=== Summary stats ===")
    print(features.describe().T)

    print("\n=== Feature means by AtRisk label (sanity check) ===")
    print(features.groupby("AtRisk").mean(numeric_only=True).T)

    features.to_parquet("data/customer_features.parquet", index=False)
    print("\nSaved: data/customer_features.parquet")