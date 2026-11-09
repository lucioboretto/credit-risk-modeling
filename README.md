# 🏦 Credit Risk Modeling

An end-to-end Data Science project that predicts the probability of default of home equity loans and turns those predictions into a credit approval policy.

---

## 📌 Project Overview

When a bank receives a loan application, it has to answer a simple question:

> **"Should we approve this loan?"**

A model that predicts who will default is only half of the answer. A probability of default needs to be combined with the money the bank earns from good clients and the money it loses from bad ones to become a decision.

This project covers the complete workflow: data understanding, cleaning, exploratory analysis, feature engineering, modeling, and the design of a credit policy based on the predicted probabilities.

---

## 🎯 Motivation

The project aims to answer three questions:

1. **Can borrower and credit history data be used to estimate the probability of default?**
2. **What approval threshold maximizes the profit of the bank?**
3. **Why does the model consider a specific client risky?**

---

## 📊 Dataset

The project uses the **HMEQ (Home Equity)** dataset, which contains information about 5,960 home equity loans. The target variable `BAD` indicates whether the client defaulted on the loan (≈20% of the loans).

| Variable | Description |
|---|---|
| BAD | 1 = client defaulted, 0 = loan repaid |
| LOAN | Amount of the loan request |
| MORTDUE | Amount due on the existing mortgage |
| VALUE | Value of the current property |
| REASON | DebtCon = debt consolidation, HomeImp = home improvement |
| JOB | Occupational category |
| YOJ | Years at present job |
| DEROG | Number of major derogatory reports |
| DELINQ | Number of delinquent credit lines |
| CLAGE | Age of the oldest credit line (months) |
| NINQ | Number of recent credit inquiries |
| CLNO | Number of credit lines |
| DEBTINC | Debt-to-income ratio |

---

## 🔍 Project Workflow

### 1. Data Understanding
Dataset structure, data types, missing values, duplicates, basic statistics and outlier detection with the IQR method.

### 2. Data Cleaning
Missing indicators (`_missing`) were created for every numerical variable with missing values, and missing categories were replaced with `Unknown`. Numerical imputation was left for the modeling pipeline to avoid data leakage.

### 3. Exploratory Data Analysis
Analysis of the relationship between each variable and default. The most relevant finding: **missing values are not random**. When `DEBTINC` is missing, the default rate is 62%, compared with 9% when it is reported.

### 4. Feature Engineering
New credit risk indicators were created:

- **LTV:** loan to value
- **MTV:** mortgage to value
- **DTV:** total debt to value
- **NEG_RECORDS:** total negative records (derogatory reports + delinquent lines)

### 5. Modeling
Three models were compared with 5-fold stratified cross-validation:

- Logistic Regression
- Random Forest
- XGBoost

Accuracy was not used due to the class imbalance. The models were evaluated with ROC-AUC, Gini, PR-AUC, KS and Brier score, and the calibration of the probabilities was checked.

### 6. Credit Policy
The predicted probabilities were used to choose an approval threshold, simulate the profit of different policies, build risk grades and explain individual decisions with SHAP.

---

## 🤖 Final Model

XGBoost was selected based on the cross-validation results.

### Performance (test set)

| Model | ROC-AUC | Gini | PR-AUC | KS | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.902 | 0.804 | 0.784 | 0.681 | 0.083 |
| Random Forest | 0.957 | 0.915 | 0.866 | 0.794 | 0.066 |
| **XGBoost** | **0.950** | **0.900** | **0.868** | **0.768** | **0.065** |

The performance is very high compared with typical credit scoring models (Gini 0.4–0.7), which is probably related to the characteristics of this dataset.

---

## 💰 Credit Policy

Assumptions: **LGD = 40%** (share of the loan lost when a client defaults) and **profit margin = 10%** (net profit from a good loan).

With these assumptions, a loan is profitable on average when **PD < 20%** (break-even point). The optimal threshold chosen on the training data was **16%**.

### Policy comparison (test set)

| Policy | Approval rate | Default rate of approved loans | Profit |
|---|---:|---:|---:|
| Approve everyone | 100% | 20.0% | 81,290 |
| Default threshold (50%) | 84.0% | 7.9% | 1,137,100 |
| **Optimal threshold (16%)** | **74.7%** | **3.7%** | **1,396,210** |

### Risk grades (test set)

| Grade | PD range | Share of clients | Observed default rate | Action |
|---|---|---:|---:|---|
| A | 0–2% | 37.2% | 0.7% | Approve |
| B | 2–5% | 22.7% | 2.2% | Approve |
| C | 5–10% | 10.4% | 12.1% | Approve |
| D | 10–20% | 6.4% | 19.7% | Approve with review |
| E | 20–50% | 7.3% | 46.0% | Reject |
| F | 50–100% | 16.0% | 83.2% | Reject |

---

## 🧠 Model Interpretability

SHAP was used to explain the model globally and for individual clients. The most important variables were:

- `DEBTINC_missing`: a missing debt-to-income ratio strongly increases the risk
- `NEG_RECORDS` and `DELINQ`: negative records in the credit history
- `CLAGE`: a long credit history reduces the risk
- `DEBTINC`: very high debt-to-income ratios increase the risk

Individual explanations (waterfall plots) show the main reasons behind each decision, which could be used as reason codes in a rejection letter.

---

## ⚠️ Limitations

- The dataset is small and has no time dimension, so the model could not be validated on more recent applications (out-of-time validation).
- The model slightly underestimates the risk in the middle grades. A calibration step would be needed before using the expected loss for provisions.
- The importance of `DEBTINC_missing` should be validated with the business to confirm that this information is available when the application is evaluated.
- Variables such as `JOB` should be reviewed for fairness before being used in real lending decisions.
- LGD and profit margin are assumptions, not real values.

---

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · SHAP · Jupyter

---

## 📁 Project Structure

```
credit-risk-modeling/
├── data/
│   ├── hmeq.csv
│   ├── HMEQ_clean.csv
│   ├── HMEQ_features.csv
│   ├── X_test.csv
│   ├── train_scores.csv
│   └── test_scores.csv
├── models/
│   └── xgboost_credit_risk_model.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_credit_policy.ipynb
├── requirements.txt
└── README.md
```
