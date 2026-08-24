# Retention Risk Scoring

A customer churn/retention-risk scoring system built end-to-end on the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) - from raw, messy transaction logs through to a deployed, explainable scoring app.

**Live app:** https://retention-risk-scoring-vz1.streamlit.app/

## The problem

Online Retail II has no churn label. It's just a receipt log: ~1.07M transactions from a UK-based online gift retailer, spanning Dec 2009–Dec 2011, with no column anywhere saying who "churned." That's the core challenge this project solves - **defining what "at risk" even means from raw behaviour, then predicting it before it happens, with reasons a non-technical stakeholder could act on.**

In plain terms: this is a system that reads a shop's sales history and tells them, in advance and with explanations, which customers are quietly about to disappear.

## Pipeline overview

```
Raw transactions (.xlsx)
        │
        ▼
1. Clean (src/load_clean.py)
        │
        ▼
2. Define inactivity window & label (src/define_label.py)
        │
        ▼
3. Engineer RFM + behavioural features (src/feature_engineering.py)
        │
        ▼
4. Train & validate model (src/modeling.py)
        │
        ▼
5. Explain with SHAP (src/shap_explainability.py)
        │
        ▼
6. Score all customers & serve via app (src/run_scoring.py, app/app.py)
```

---

## Phase 1 - Cleaning

The raw data has real messiness: cancelled invoices (prefixed `C`), missing `CustomerID` on ~20-25% of rows, non-product line items (postage, bank charges), and mixed data types in ID-like columns.

**Decisions made, and why:**

- **Dropped rows with missing `CustomerID`.** Can't attribute a transaction to a customer for a per-customer risk model, so these are unusable here. This is a real bias worth naming: it likely skews the dataset toward registered/attributed customers and away from guest-style checkouts.
- **Kept cancellations as a flag rather than dropping them.** A customer who cancels frequently is itself a potential risk signal - discarding that information would throw away something the model could learn from.
- **Dropped non-product stock codes** (`POST`, `BANK CHARGES`, `D`, `M`, etc.) - these pollute monetary totals if left in, since they're not real product purchases.
- **Forced `Invoice` and `StockCode` to string type explicitly.** These columns mix numeric-looking values with alphanumeric codes (`489434` vs `C489449`), which pandas stores as `object` dtype - fine for pandas, but PyArrow's parquet writer tries to infer a single type and throws `ArrowInvalid` when it hits the mismatch. Forcing to string upfront avoids this.

**Result:** ~794K clean rows from 5,895 customers.

---

## Phase 2 - Defining "at risk"

This is the one genuinely judgment-call part of the whole project, so it's derived from evidence rather than picked as a round number.

**Method:**
1. Computed the gap in days between every customer's consecutive purchases, pooled across all 5,895 customers.
2. Set a snapshot date at 65% through the timeline (26 March 2011) - not the very end of the dataset - so there's a genuine holdout tail of real future data to validate against.
3. Swept multiple candidate windows (60/90/120/130/150/180/200/210 days) and, for each, checked:
   - **False alarm rate** - of customers flagged at-risk, what % actually came back anyway (want low)
   - **Recall of actual churners** - of customers who genuinely went quiet, what % did the window catch (want high)

**Finding:** the obvious-looking metric (best "separation") pointed toward larger windows (180-210 days), but that was misleading - recall collapsed from 86% at 60 days to just 41% at 210 days. A window that looks clean on paper was silently missing most real churners.

**Chosen window: 120 days** - the point where recall (73%) and false-alarm rate (40%) are both reasonable, just before recall starts falling faster than precision improves. This is documented as a genuine trade-off, not a "solved" number.

---

## Phase 3 - Feature engineering

Built a per-customer feature table using **only pre-snapshot transactions** - the same discipline as the label, to avoid leaking future information into the features.

| Feature | What it captures |
|---|---|
| Frequency | Number of distinct orders |
| Monetary | Total spend |
| Tenure | Days since first purchase |
| AvgOrderValue, AvgBasketSize | Typical order size/value |
| UniqueProducts | Breadth of what they buy |
| SpendTrend | Whether spending is rising or falling, comparing the early vs. late half of their own history |

**Bug caught and fixed:** the first version of `SpendTrend` divided by `(early + late)` spend, which can sit near zero when cancellations cause the two periods to nearly cancel out — this blew the ratio up to absurd values (seen: -4.3 trillion mean). Fixed by dividing by `abs(early) + abs(late)` instead, which stays safely bounded in [-1, 1].

**Known limitation, left as-is:** `Monetary`, `AvgOrderValue`, and `AvgBasketSize` can go negative for customers whose cancellations outweighed purchases pre-snapshot. This is real, not a bug - just worth knowing if extending the model.

---

## Phase 4 - Modeling

Trained a Logistic Regression baseline and an XGBoost classifier.

**The leakage catch:** the first modeling run scored a suspicious **100% accuracy, ROC-AUC 1.000** on both models, with `Recency` alone carrying 75% of XGBoost's feature importance. The cause: `Recency` (days since last purchase, as a feature) is near-identical to the exact quantity used to *define* the `AtRisk` label in Phase 2. The model wasn't predicting anything - it was reading the label's own definition back to itself.

**Fix:** dropped `Recency` from the feature set entirely and retrained.

**Honest final results:**

| Model | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.869 | 0.857 | 82% |
| **XGBoost (chosen)** | **0.926** | **0.938** | **84%** |

Balanced precision/recall across both classes (~0.84-0.85), with a believable error rate (72 false negatives, 76 false positives out of 943 test customers) - the kind of result that holds up to scrutiny, unlike the leaked version.

**What actually drives predictions**, per XGBoost feature importance: Frequency (39%), Tenure (23%), Monetary (13%) - behavioural loyalty signals rather than raw spend.

---

## Phase 5 - Explainability (SHAP)

Global SHAP values confirm the same story as feature importance, but the real value is per-customer explanations. Three illustrative cases from the test set:

- **Confident true positive** (Customer 18174, risk 1.00): bought once, spent very little - a textbook one-and-done customer, correctly caught.
- **False positive** (Customer 17171, risk 0.90): long tenure (402 days) but very low frequency (2 orders). The model reads "rarely buys" as risky, but this looks like a naturally infrequent, loyal buyer rather than someone leaving. Worth naming as a known model tendency.
- **False negative** (Customer 12835, risk 0.02): a genuinely high-value, frequent customer (49 orders, £5,972 spent) who nonetheless went quiet. The model trusted their strong history and missed a real behavioural shift - exactly the case `SpendTrend` was designed to catch, and a natural next-iteration target (e.g. weighting recent behaviour more heavily).

---

## Phase 6 - Deployment

- **`src/run_scoring.py`** - regenerates features and SHAP-based reasons for every customer, reusing the exact Phase 3 functions (not a re-implementation) to avoid train/serve skew.
- **`app/app.py`** (Streamlit) - look up any customer's risk score, prediction, and top 3 plain-English reasons, or browse a leaderboard of the highest-risk customers with a risk distribution chart.
- **Re-scoring is manual, by design.** A scheduled GitHub Actions workflow was built and tested, but removed: it depended on a static committed data file rather than a live data source, which doesn't reflect how this would actually work in production. Re-scoring is run on demand instead:
  ```
  python src/run_scoring.py
  ```

---

## Tech stack

Python, pandas, numpy, scikit-learn, XGBoost, SHAP, Streamlit, PyArrow (parquet), matplotlib.

## Setup

```bash
git clone https://github.com/rajdeepgupta68/retention-risk-scoring.git
cd retention-risk-scoring
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download `online_retail_II.xlsx` from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) into `data/`, then run the pipeline in order:

```bash
python src/load_clean.py
python src/define_label.py
python src/feature_engineering.py
python src/modeling.py
python src/shap_explainability.py
python src/run_scoring.py
streamlit run app/app.py
```

## Known limitations

- ~20-25% of raw transactions were dropped for missing `CustomerID`, introducing a real (unmeasured) bias toward attributed/registered customers.
- The 120-day inactivity window is a defensible trade-off, not a universal answer - a different retail context would likely need re-deriving it.
- The model tends to over-flag long-tenure, low-frequency customers, and can under-flag customers with a strong history but a recent behavioural shift.
- Re-scoring is a manual, on-demand script rather than a live-data-connected scheduled job.

## Possible next steps

- Weight recent behaviour more heavily (e.g. a shorter-window recency-of-frequency feature) to catch drift like the false-negative case above.
- Connect `run_scoring.py` to a real data source instead of a static file, and reinstate scheduled re-scoring.
- Tune the classification threshold based on business cost (a missed churner vs. a wasted retention email likely aren't equally costly).
