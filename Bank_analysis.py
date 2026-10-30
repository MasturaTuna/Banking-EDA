# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:56:25 2026

@author: Admin
"""

"""
Banking term-deposit case: EDA (Q1-Q18) + simple predictive model.

Usage:
    python banking_eda.py path/to/banking_data.csv [output_dir]

Outputs (in output_dir, default ./outputs):
    fig_*.png          one chart per question
    summary.txt        all printed numbers (same as console)
    model_*.png        ROC curve, gains curve, feature importance
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, precision_recall_curve,
                             roc_auc_score, roc_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
CSV = sys.argv[1] if len(sys.argv) > 1 else "banking_data.csv"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs")
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=130)
    plt.close()


def pct(s):
    return (s.value_counts(normalize=True) * 100).round(2)


# ----------------------------------------------------------------------------
# 0. Load & clean
# ----------------------------------------------------------------------------
raw = pd.read_csv(CSV)
say("=== 0. DATA QUALITY ===")
say("Raw shape:", raw.shape, "(brief says 45,211 x 18; file has 2 extra helper cols:"
    " marital_status, day_month)")
say("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0].to_string())
say("Exact duplicate rows:", raw.duplicated().sum())
say("marital vs marital_status mismatches (excluding NaN):",
    ((raw.marital != raw.marital_status) & raw.marital.notna()).sum())

df = raw.drop_duplicates().dropna(subset=["marital", "education"]).copy()
df = df.drop(columns=["marital_status", "day_month"])  # redundant copies
df["target"] = (df["y"] == "yes").astype(int)
say("Cleaned shape:", df.shape, "-> rows removed:", len(raw) - len(df))
say("'unknown' is kept as its own category (job/education/contact/poutcome);"
    " it is informative, not random missingness.\n")

# ----------------------------------------------------------------------------
# Q1 Age
# ----------------------------------------------------------------------------
say("=== Q1 AGE ===")
say(df.age.describe().round(2).to_string())
say("Skewness:", round(df.age.skew(), 2), "| outliers (>1.5 IQR):",
    int(((df.age > df.age.quantile(.75) + 1.5 * stats.iqr(df.age))).sum()))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.age, bins=40, kde=True, ax=ax[0])
sns.boxplot(x=df.age, ax=ax[1])
ax[0].set_title("Q1 Age distribution")
save("fig_q01_age")

# ----------------------------------------------------------------------------
# Q2-Q4 Job, marital, education
# ----------------------------------------------------------------------------
for q, col in [(2, "job"), (3, "marital"), (4, "education")]:
    say(f"\n=== Q{q} {col.upper()} (% of clients) ===")
    say(pct(df[col]).to_string())
    order = df[col].value_counts().index
    plt.figure(figsize=(9, 4.5))
    sns.countplot(y=df[col], order=order, color="#3b6ea5")
    plt.title(f"Q{q} {col} distribution")
    save(f"fig_q{q:02d}_{col}")

# ----------------------------------------------------------------------------
# Q5 Default
# ----------------------------------------------------------------------------
say("\n=== Q5 CREDIT IN DEFAULT ===")
say(df.default.value_counts().to_string())
say(pct(df.default).to_string())

# ----------------------------------------------------------------------------
# Q6 Balance
# ----------------------------------------------------------------------------
say("\n=== Q6 BALANCE (EUR) ===")
say(df.balance.describe().round(1).to_string())
say("Negative balances:", int((df.balance < 0).sum()),
    f"({(df.balance < 0).mean():.1%}) | skew:", round(df.balance.skew(), 2))
say("Balance quantiles 1/5/50/95/99%:",
    df.balance.quantile([.01, .05, .5, .95, .99]).round(0).tolist())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.balance.clip(-2000, 10000), bins=60, ax=ax[0])
ax[0].set_title("Q6 Balance (clipped to -2k..10k for readability)")
sns.histplot(np.sign(df.balance) * np.log1p(df.balance.abs()), bins=60, ax=ax[1])
ax[1].set_title("Signed log(balance)")
save("fig_q06_balance")

# ----------------------------------------------------------------------------
# Q7-Q9 Housing, loan, contact
# ----------------------------------------------------------------------------
for q, col in [(7, "housing"), (8, "loan"), (9, "contact")]:
    say(f"\n=== Q{q} {col.upper()} ===")
    say(pd.DataFrame({"n": df[col].value_counts(), "%": pct(df[col])}).to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, c in zip(ax, ["housing", "loan", "contact"]):
    sns.countplot(x=df[c], ax=a, color="#3b6ea5")
    a.set_title(c)
fig.suptitle("Q7-Q9 Housing loan, personal loan, contact type")
save("fig_q07_09_loans_contact")

# ----------------------------------------------------------------------------
# Q10 Day, Q11 Month
# ----------------------------------------------------------------------------
say("\n=== Q10 DAY OF MONTH ===")
say(df.day.describe().round(2).to_string())
say("Busiest 5 days:", df.day.value_counts().head(5).to_dict())
plt.figure(figsize=(10, 4))
sns.countplot(x=df.day, color="#3b6ea5")
plt.title("Q10 Last contact day of month")
save("fig_q10_day")

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
say("\n=== Q11 MONTH ===")
mc = df.month.value_counts().reindex(months)
say(pd.DataFrame({"n": mc, "%": (mc / mc.sum() * 100).round(2)}).to_string())
plt.figure(figsize=(10, 4))
sns.countplot(x=df.month, order=months, color="#3b6ea5")
plt.title("Q11 Last contact month")
save("fig_q11_month")

# ----------------------------------------------------------------------------
# Q12 Duration
# ----------------------------------------------------------------------------
say("\n=== Q12 DURATION (sec) ===")
say(df.duration.describe().round(1).to_string())
say("Skew:", round(df.duration.skew(), 2), "| calls < 10s:", int((df.duration < 10).sum()),
    "| calls > 1h:", int((df.duration > 3600).sum()))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.duration.clip(upper=1500), bins=60, ax=ax[0])
ax[0].set_title("Q12 Duration (clipped at 1500s)")
sns.boxplot(x=df.duration, ax=ax[1])
save("fig_q12_duration")

# ----------------------------------------------------------------------------
# Q13 Campaign, Q14 pdays, Q15 previous, Q16 poutcome
# ----------------------------------------------------------------------------
say("\n=== Q13 CAMPAIGN (contacts this campaign) ===")
say(df.campaign.describe().round(2).to_string())
say("Share with 1/2/3 contacts:", pct(df.campaign).head(3).to_dict(),
    "| >10 contacts:", int((df.campaign > 10).sum()))

say("\n=== Q14 PDAYS ===")
never = (df.pdays == -1)
say(f"Never contacted before (pdays=-1): {never.sum()} ({never.mean():.1%})")
say("Among previously contacted:\n", df.loc[~never, "pdays"].describe().round(1).to_string())

say("\n=== Q15 PREVIOUS ===")
say(df.previous.describe().round(2).to_string())
say(f"previous=0: {(df.previous == 0).mean():.1%} | max: {df.previous.max()}"
    " (275 is an extreme outlier)")

say("\n=== Q16 POUTCOME ===")
say(pd.DataFrame({"n": df.poutcome.value_counts(), "%": pct(df.poutcome)}).to_string())
say(f"Among previously contacted only, success rate: "
    f"{(df.loc[df.previous > 0, 'poutcome'] == 'success').mean():.1%}")

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
sns.histplot(df.campaign.clip(upper=20), bins=20, ax=ax[0, 0]); ax[0, 0].set_title("Q13 campaign (clip 20)")
sns.histplot(df.loc[~never, "pdays"], bins=50, ax=ax[0, 1]); ax[0, 1].set_title("Q14 pdays (previously contacted only)")
sns.histplot(df.previous.clip(upper=15), bins=15, ax=ax[1, 0]); ax[1, 0].set_title("Q15 previous (clip 15)")
sns.countplot(x=df.poutcome, ax=ax[1, 1], color="#3b6ea5"); ax[1, 1].set_title("Q16 poutcome")
save("fig_q13_16_campaign_history")

# ----------------------------------------------------------------------------
# Q17 Target distribution
# ----------------------------------------------------------------------------
say("\n=== Q17 TARGET (subscribed term deposit) ===")
say(pd.DataFrame({"n": df.y.value_counts(), "%": pct(df.y)}).to_string())
say(f"Class imbalance ratio no:yes = {(df.target == 0).sum() / df.target.sum():.1f}:1")
plt.figure(figsize=(5, 4))
sns.countplot(x=df.y, color="#3b6ea5")
plt.title("Q17 Subscription (y)")
save("fig_q17_target")

# ----------------------------------------------------------------------------
# Q18 Relationships with subscription
# ----------------------------------------------------------------------------
say("\n=== Q18 WHAT RELATES TO SUBSCRIPTION? ===")
base = df.target.mean()
say(f"Baseline subscription rate: {base:.2%}")

cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
rows = []
for c in cat_cols:
    ct = pd.crosstab(df[c], df.target)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    rows.append((c, round(cramers_v, 3), p))
    rate = (df.groupby(c).target.mean() * 100).round(1).sort_values(ascending=False)
    say(f"\n[{c}] conversion % by level (Cramer's V={cramers_v:.3f}, p={p:.1e}):")
    say(rate.to_string())
cv = pd.DataFrame(rows, columns=["feature", "cramers_v", "p_value"]).sort_values("cramers_v", ascending=False)
say("\nCategorical association strength ranking:\n", cv.to_string(index=False))

num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
say("\nNumeric: point-biserial correlation with target + mean by class")
pb = []
for c in num_cols:
    r, p = stats.pointbiserialr(df.target, df[c])
    pb.append((c, round(r, 3), p, round(df.loc[df.target == 1, c].mean(), 1),
               round(df.loc[df.target == 0, c].mean(), 1)))
say(pd.DataFrame(pb, columns=["feature", "r", "p", "mean_yes", "mean_no"]).to_string(index=False))

# Binned views (more informative than raw correlation for skewed vars)
df["age_band"] = pd.cut(df.age, [0, 24, 34, 44, 54, 64, 100],
                        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
df["balance_band"] = pd.qcut(df.balance, 5, duplicates="drop")
df["duration_band"] = pd.cut(df.duration, [-1, 60, 120, 240, 480, 900, 6000],
                             labels=["<1m", "1-2m", "2-4m", "4-8m", "8-15m", "15m+"])
df["campaign_band"] = pd.cut(df.campaign, [0, 1, 2, 3, 5, 10, 100],
                             labels=["1", "2", "3", "4-5", "6-10", "11+"])
for c in ["age_band", "balance_band", "duration_band", "campaign_band"]:
    say(f"\nConversion % by {c}:")
    say((df.groupby(c, observed=True).target.mean() * 100).round(1).to_string())

fig, ax = plt.subplots(3, 3, figsize=(16, 12))
for a, c in zip(ax.ravel(), ["job", "marital", "education", "contact", "month", "poutcome",
                              "age_band", "duration_band", "campaign_band"]):
    order = months if c == "month" else None
    rate = df.groupby(c, observed=True).target.mean() * 100
    if order:
        rate = rate.reindex(order)
    elif c not in ("age_band", "duration_band", "campaign_band"):
        rate = rate.sort_values()
    rate.plot.barh(ax=a, color="#3b6ea5") if c not in ("age_band", "duration_band", "campaign_band", "month") \
        else rate.plot.bar(ax=a, color="#3b6ea5")
    a.axhline(base * 100, color="red", ls="--", lw=1) if c in ("age_band", "duration_band", "campaign_band", "month") \
        else a.axvline(base * 100, color="red", ls="--", lw=1)
    a.set_title(f"Conversion % by {c} (red = {base:.1%} baseline)")
    a.set_xlabel(""); a.set_ylabel("")
save("fig_q18_conversion_by_feature")

plt.figure(figsize=(8, 6))
corr = df[num_cols + ["target"]].corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Q18 Spearman correlation (numeric + target)")
save("fig_q18_corr_heatmap")

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
say("\n\n=== MODEL: who will subscribe? ===")
say("`duration` is only known AFTER the call, so it cannot be used to decide who to call."
    " Main model excludes it; a with-duration model is shown only as a reference ceiling.")

df["was_contacted_before"] = (df.pdays != -1).astype(int)
cats = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
nums = ["age", "balance", "day", "campaign", "pdays", "previous", "was_contacted_before"]


def prep(num, cat):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def fit_eval(features_num, features_cat, label):
    X = df[features_num + features_cat]
    y = df.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "LogReg": Pipeline([("p", prep(features_num, features_cat)),
                            ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "GradBoost": Pipeline([("p", prep(features_num, features_cat)),
                               ("m", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                                    max_iter=300, random_state=42))]),
    }
    res = {}
    for n, m in models.items():
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        res[n] = (m, pr)
        say(f"[{label}] {n}: ROC-AUC={roc_auc_score(yte, pr):.3f}  "
            f"PR-AUC={average_precision_score(yte, pr):.3f}")
    return models, res, Xte, yte


# main model (deployable)
models, res, Xte, yte = fit_eval(nums, cats, "no duration")
# reference ceiling
fit_eval(nums + ["duration"], cats, "WITH duration (leaky ceiling)")

best = "GradBoost"
m, pr = res[best]

# ROC
plt.figure(figsize=(6, 5))
for n, (_, p) in res.items():
    fpr, tpr, _ = roc_curve(yte, p)
    plt.plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(yte, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC - deployable model (no duration)")
save("model_roc")

# Gains / lift by decile -> the business answer for "who to call"
g = pd.DataFrame({"y": yte.values, "p": pr}).sort_values("p", ascending=False).reset_index(drop=True)
g["decile"] = pd.qcut(g.index, 10, labels=range(1, 11))
tab = g.groupby("decile", observed=True).y.agg(["count", "sum", "mean"])
tab["cum_capture_%"] = (tab["sum"].cumsum() / tab["sum"].sum() * 100).round(1)
tab["lift"] = (tab["mean"] / g.y.mean()).round(2)
tab["mean"] = (tab["mean"] * 100).round(1)
tab.columns = ["clients", "subscribers", "conv_%", "cum_capture_%", "lift"]
say("\nDecile gains table (decile 1 = highest predicted probability):\n", tab.to_string())
plt.figure(figsize=(6, 5))
plt.plot(np.arange(0, 11) * 10, [0] + tab["cum_capture_%"].tolist(), marker="o", label="Model")
plt.plot([0, 100], [0, 100], "k--", label="Random calling")
plt.xlabel("% of clients called (ranked by model)"); plt.ylabel("% of subscribers captured")
plt.legend(); plt.title("Cumulative gains")
save("model_gains")

# threshold guidance
prec, rec, thr = precision_recall_curve(yte, pr)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
t = thr[np.argmax(f1[:-1])]
say(f"\nF1-optimal threshold={t:.3f}")
say(classification_report(yte, (pr >= t).astype(int), target_names=["no", "yes"], digits=3))

# permutation importance (on raw features so one-hot levels aren't split)
pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
say("\nPermutation importance (drop in ROC-AUC):\n", imp.sort_values(ascending=False).round(4).to_string())
plt.figure(figsize=(7, 5))
imp.plot.barh(color="#3b6ea5")
plt.title("Feature importance (permutation, ROC-AUC drop)")
save("model_importance")

# logistic regression odds ratios for interpretability
lr = models["LogReg"].named_steps["m"]
names = models["LogReg"].named_steps["p"].get_feature_names_out()
odds = pd.Series(np.exp(lr.coef_[0]), index=names).sort_values(ascending=False)
say("\nLogReg odds ratios - top 10 up / down (standardised numerics, one-hot cats):")
say(odds.head(10).round(2).to_string()); say(odds.tail(10).round(2).to_string())

say("\nDone. Figures + summary.txt written to", OUT.resolve())
LOG.close()