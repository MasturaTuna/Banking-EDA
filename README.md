
# Banking Term-Deposit Analysis & Predictive Modeling

This repository contains an end-to-end Python pipeline (`Bank_analysis.py`) designed to perform Exploratory Data Analysis (EDA) and train predictive models on bank telemarketing campaign data. The primary objective is to identify key factors influencing client subscriptions to term deposits and to build a deployable classification model for targeting potential subscribers.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Specifications](#dataset-specifications)
- [Project Workflow](#project-workflow)
  - [1. Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [2. Exploratory Data Analysis (Q1–Q18)](#2-exploratory-data-analysis-q1q18)
  - [3. Predictive Modeling](#3-predictive-modeling)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Generated Outputs](#generated-outputs)
- [Key Business Insights & Strategy](#key-business-insights--strategy)

---

## 🔍 Overview

The script automates the complete data science process for bank telemarketing data:
1. **Data Sanitization**: Removes duplicates, resolves redundant columns, and handles missing/unknown values.
2. **Exploratory Analysis (Q1–Q18)**: Analyzes single-variable distributions, missing patterns, balance anomalies, and statistical associations (Cramer's V, point-biserial correlation, Spearman heatmap).
3. **Machine Learning Pipeline**: Trains and evaluates classifiers (**Logistic Regression** and **HistGradientBoostingClassifier**) using a leak-free setup (excluding post-call features like call duration for actual deployment).

---

## 📊 Dataset Specifications

The expected input CSV file (default: `banking_data.csv`) includes client demographics, financial metrics, and campaign interaction history:

* **Demographics**: `age`, `job`, `marital`, `education`
* **Financials**: `default` (credit in default), `balance` (average yearly balance in EUR), `housing` (housing loan), `loan` (personal loan)
* **Campaign Details**: `contact` (communication type), `day` (last contact day), `month` (last contact month), `duration` (last contact duration in seconds)
* **Interaction History**: `campaign` (number of contacts during this campaign), `pdays` (days since previous campaign contact), `previous` (number of contacts before this campaign), `poutcome` (outcome of previous marketing campaign)
* **Target Variable**: `y` (has the client subscribed a term deposit? `"yes"` / `"no"`)

---

## ⚙️ Project Workflow

### 1. Data Cleaning & Preprocessing
* Removes exact duplicate rows.
* Drops redundant helper columns (`marital_status`, `day_month`).
* Drops rows with missing critical attributes (`marital`, `education`).
* Preserves `"unknown"` as a distinct category across categorical attributes (`job`, `education`, `contact`, `poutcome`), as missingness in marketing data often carries predictive signal.

### 2. Exploratory Data Analysis (Q1–Q18)
The script systematically evaluates 18 questions/dimensions:
* **Q1**: Age distribution, skewness, and IQR outlier detection.
* **Q2–Q4**: Categorical distributions for Job, Marital Status, and Education.
* **Q5**: Credit default rate analysis.
* **Q6**: Balance analysis including negative balances, extreme quantiles, and signed-log transformations.
* **Q7–Q9**: Housing loans, personal loans, and contact channels.
* **Q10–Q11**: Temporal trends (busiest days of the month and months of the year).
* **Q12**: Call duration distributions and outlier checks.
* **Q13–Q16**: Prior contact metrics (`campaign`, `pdays`, `previous`, `poutcome`).
* **Q17**: Baseline target subscription rate and class imbalance ratio.
* **Q18**: Feature association analysis using Cramer's V (categorical), Point-Biserial correlation (numerical), Spearman rank correlations, and binned conversion rates (`age_band`, `balance_band`, `duration_band`, `campaign_band`).

### 3. Predictive Modeling
* **Data Leakage Mitigation**: Excludes `duration` from the primary deployable model since call duration is unknown prior to making a call. A secondary model including `duration` is evaluated only as a benchmark.
* **Feature Engineering**: Adds indicator `was_contacted_before` ($pdays \neq -1$).
* **Pipelines**: Standard scaling for numeric features; One-Hot Encoding for categoricals.
* **Models**:
  * **Logistic Regression**: Serves as an interpretable baseline (evaluates odds ratios).
  * **HistGradientBoostingClassifier**: Non-linear tree model optimized for tabular data.
* **Evaluation Metrics**: ROC-AUC, Precision-Recall AUC (PR-AUC), Decile Gains/Lift Table, and Permutation Feature Importance.

---

## 💻 Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

---

## 🚀 Usage

Execute the script from the command line by providing the path to your CSV dataset and an optional output directory.

### Basic Syntax
```bash
python Bank_analysis.py <path_to_csv> [output_directory]
```

### Examples

**Default execution** (looks for `banking_data.csv` in the current working directory and outputs to `./outputs`):
```bash
python Bank_analysis.py
```

**Custom CSV and custom output directory**:
```bash
python Bank_analysis.py data/bank_marketing.csv results/eda_2026
```

---

## 📁 Generated Outputs

All outputs are saved to the designated output folder (`./outputs` by default):

| File Name | Description |
| :--- | :--- |
| `summary.txt` | Complete text log containing data summary statistics, chi-square tests, correlation tables, model metrics, and decile gains tables. |
| `fig_q01_age.png` | Histogram with KDE and Boxplot of age distribution. |
| `fig_q02_job.png` to `fig_q04_education.png` | Count plots of client demographics. |
| `fig_q06_balance.png` | Histogram of raw (clipped) and signed log-transformed balances. |
| `fig_q07_09_loans_contact.png` | Multi-panel distribution of loans and contact types. |
| `fig_q10_day.png` / `fig_q11_month.png` | Temporal contact distribution plots. |
| `fig_q12_duration.png` | Distribution and boxplot of call duration. |
| `fig_q13_16_campaign_history.png` | 2x2 grid visualizing prior campaign metrics and outcome distributions. |
| `fig_q17_target.png` | Class imbalance visualization for the target variable `y`. |
| `fig_q18_conversion_by_feature.png` | 3x3 horizontal/vertical bar charts showing conversion rates vs. baseline across features. |
| `fig_q18_corr_heatmap.png` | Spearman correlation matrix heatmap across numeric features and target. |
| `model_roc.png` | Receiver Operating Characteristic (ROC) curve comparison across models. |
| `model_gains.png` | Cumulative gains chart depicting percentage of subscribers captured per decile targeted. |
| `model_importance.png` | Permutation feature importance chart (ROC-AUC drop metric). |

---

## 💡 Key Business Insights & Strategy

* **Call Targeting (Decile Lift)**: By ranking prospective clients using the **HistGradientBoosting** deployable model, the bank can capture a majority of potential term-deposit subscribers while contacting only a fraction of the total client base.
* **Leakage Awareness**: Call `duration` is the single strongest indicator of subscription success, but it cannot be used for lead generation planning. Pre-call attributes like `poutcome`, `month`, `housing`, and `age` drive primary lead selection.
* **Prior Campaign Success**: Clients who responded positively in previous campaigns (`poutcome == 'success'`) show significantly higher conversion rates, making them top candidates for re-engagement.
