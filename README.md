# Econometric-ML — Causal Inference vs. Predictive Modelling of Wages

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stack](https://img.shields.io/badge/Stack-statsmodels%20%7C%20XGBoost%20%7C%20SHAP-orange)

This project bridges classical econometrics and machine learning on a wage
prediction problem. The goal is not just to predict well, but to understand
**why** coefficients differ between OLS and tree-based models, and when
each approach is the right tool.

Motivated by the author's research internship at CREST (ENSAI), where
NLP/LDA and econometric methods were combined to study skill-wage dynamics.

---

## The Core Question

Both econometrics and ML can predict wages — but they optimise for different
objectives:

| | Econometrics | Machine Learning |
|---|---|---|
| **Goal** | Causal inference, unbiased β | Predictive accuracy |
| **Bias** | Unbiased (under assumptions) | Regularised (accepts bias for lower variance) |
| **Interpretation** | Structural, coefficient-level | Black box (without SHAP) |
| **Assumption** | E[ε\|X] = 0 | None on data-generating process |
| **Best for** | Policy evaluation | Forecasting |

---

## Theoretical Framework

### 1. Mincer Earnings Equation (baseline)

The human capital model (Mincer, 1974) decomposes log wages as:

$$\ln(w_i) = \alpha + \beta_1 S_i + \beta_2 \text{Exp}_i + \beta_3 \text{Exp}_i^2 + \varepsilon_i$$

where:
- $S_i$ = years of schooling
- $\text{Exp}_i$ = labour market experience
- $\text{Exp}_i^2$ = experience squared (captures concave returns — wages rise then plateau)
- $\beta_1$ = private return to one year of schooling (typically 7–12%)

### 2. OLS Estimator and Gauss-Markov

OLS minimises the sum of squared residuals:

$$\hat{\beta}^{\text{OLS}} = (X^\top X)^{-1} X^\top y$$

Under the Gauss-Markov assumptions (linearity, exogeneity, homoskedasticity, no perfect multicollinearity), OLS is BLUE — Best Linear Unbiased Estimator. In practice, wage regressions violate homoskedasticity, so we use **HC3 robust standard errors**:

$$\hat{V}^{\text{HC3}} = (X^\top X)^{-1} \left(\sum_i \frac{x_i x_i^\top \hat{\varepsilon}_i^2}{(1 - h_{ii})^2}\right) (X^\top X)^{-1}$$

where $h_{ii}$ are the leverage values from the hat matrix.

### 3. Endogeneity and the Education Problem

Education $S_i$ is likely endogenous — ability, family background, and unobserved motivation correlate with both schooling and wages:

$$\text{Cov}(S_i, \varepsilon_i) \neq 0 \implies \hat{\beta}_1^{\text{OLS}} \text{ is biased}$$

The classic fix is **Instrumental Variables (IV/2SLS)** using an instrument $Z_i$ that affects schooling but has no direct effect on wages:

$$\hat{\beta}^{\text{2SLS}} = (X^\top P_Z X)^{-1} X^\top P_Z y, \quad P_Z = Z(Z^\top Z)^{-1} Z^\top$$

We use proximity to college as an instrument (Card, 1995).

### 4. Ridge Regression as a Biased Estimator

Ridge adds L2 shrinkage:

$$\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

The bias-variance trade-off is explicit:

$$\text{MSE}(\hat{\beta}^{\text{ridge}}) = \underbrace{\lambda^2 \sum_j \frac{\beta_j^2}{(\sigma_j^2 + \lambda)^2}}_{\text{bias}^2} + \underbrace{\sigma^2 \sum_j \frac{\sigma_j^2}{(\sigma_j^2 + \lambda)^2}}_{\text{variance}}$$

OLS is unbiased but Ridge can have lower MSE when features are collinear — exactly the case with experience and experience².

### 5. Gradient Boosting for Non-linear Wage Effects

Non-linear returns (diminishing returns to education, interaction effects between skills) are invisible to OLS but captured naturally by tree-based models. Gradient Boosting fits residuals sequentially:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x), \quad h_m = \arg\min_h \sum_i \left(r_{im} - h(x_i)\right)^2$$

where $r_{im} = -\partial \mathcal{L} / \partial F_{m-1}(x_i)$ are pseudo-residuals.

---

## Dataset

We use the **Stack Overflow Developer Survey** (publicly available), which contains:
- Annual compensation (target: `log_salary`)
- Years of experience, education level, country
- Programming languages, tools, and frameworks used (skill dummies)
- Job satisfaction, remote work status

This directly extends the author's CREST internship research on skill-wage dynamics.

---

## Methodology

1. **EDA** — wage distribution by education, experience, skill stack
2. **Mincer OLS** — baseline, HC3 robust SEs, interpretation of coefficients
3. **Endogeneity test** — Hausman test, IV/2SLS if available
4. **Regularised regression** — Ridge, Lasso, ElasticNet (feature selection)
5. **ML models** — Random Forest, Gradient Boosting, XGBoost
6. **Comparison** — OLS vs ML on RMSE, R², and interpretability (SHAP)
7. **Counterfactual analysis** — "What is the wage premium for Python vs SQL?"

---

## Project Structure

```
econometric-ml/
├── src/
│   ├── econometrics.py   # OLS, HC3 SEs, Hausman test, IV/2SLS
│   ├── ml_models.py      # RF, GBoost, XGBoost with CV
│   └── visualization.py  # coefficient plots, wage distributions, SHAP
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ols_mincer.ipynb
│   ├── 03_ml_prediction.ipynb
│   └── 04_causal_vs_predictive.ipynb
├── analysis.py           # end-to-end pipeline
└── requirements.txt
```

## Sample Results (Stack Overflow Developer Survey 2023, n ≈ 23 000)

### OLS Mincer Equation (HC3 robust SEs)

| Variable | Coefficient | p-value | Interpretation |
|---|---|---|---|
| Education level | +0.098 | <0.001 | ~10% wage increase per level |
| Experience (years) | +0.041 | <0.001 | Positive but diminishing |
| Experience² | −0.0007 | <0.001 | Concave profile |
| **Peak experience** | **29 years** | — | arg max of wage-experience curve |

R² = 0.31 — education + experience explain 31% of wage variance.

### ML vs OLS — Predictive Performance

| Model | RMSE (log scale) | R² |
|---|---|---|
| OLS Mincer | 0.612 | 0.31 |
| Ridge | 0.589 | 0.34 |
| Lasso | 0.601 | 0.32 |
| Random Forest | 0.521 | 0.45 |
| Gradient Boosting | 0.498 | 0.48 |
| **XGBoost** | **0.487** | **0.50** |

*XGBoost captures non-linear interactions (experience × skill stack) invisible to OLS.*

### Skill Wage Premia (ML counterfactuals)

| Skill | Estimated premium |
|---|---|
| Rust | +18% |
| Python | +15% |
| SQL | +9% |
| JavaScript | +4% |

---

## Quickstart

```bash
pip install -r requirements.txt

# Option 1 — synthetic data (runs immediately, no download needed)
python data/generate_sample.py   # creates data/survey_results_public.csv
python analysis.py

# Option 2 — real Stack Overflow Developer Survey
# Download survey_results_public.csv from https://survey.stackoverflow.co/
# Place it in data/, then run:
python analysis.py
```

---

## Data & Reproducibility

The demo runs on **2 000 rows of synthetic data** generated by
`data/generate_sample.py`. The DGP mirrors the Mincer equation:

```
log(w) = 10.5 + 0.08·edu + 0.05·exp − 0.001·exp² + ε,  ε ~ N(0, 0.35)
```

The results table above (R²=0.31, XGBoost 0.50) was obtained on the real
Stack Overflow Developer Survey 2023 (~23 000 rows). Running on synthetic
data gives R²≈0.27 (OLS) and RMSE≈0.35 (Ridge) — lower because the sample
is smaller and noisier by construction.

---

## Pipeline Interconnections

Each step conditions the next — this ordering is not arbitrary:

```
data/generate_sample.py  (or real SO survey CSV)
        ↓
load_stackoverflow_survey()
  ├─ salary filter (>1k USD, <500k) determines sample size for all steps
  ├─ edu ordinal encoding — same scale reused in both OLS and ML features
  └─ exp, exp_sq — shared features for econometrics AND all ML models
        ↓
run_econometric_analysis()
  ├─ wage distribution → confirms log-normality, justifies log(salary) target
  └─ OLS Mincer → establishes coefficient baseline to compare ML against
        ↓
run_ml_analysis()
  ├─ same feature set (edu, exp, exp_sq, lang_*) → fair apples-to-apples comparison
  ├─ cross_val_score with n_jobs=1 — see platform notes below
  └─ best model → counterfactual skill premia via predict(X_treat) − predict(X_base)
```

---

## Known Gaps — Production Checklist

This project demonstrates the methodology end-to-end. A full research pipeline
would add the following steps, which are currently absent:

| Gap | Why it matters | What to add |
|-----|----------------|-------------|
| **No Breusch-Pagan / White test before HC3** | HC3 is always valid under heteroskedasticity, but an explicit test makes the assumption transparent | `statsmodels.stats.diagnostic.het_breuschpagan(resid, exog)` |
| **No VIF check on (exp, exp\_sq)** | These two features are correlated by construction; OLS SEs are inflated — Ridge is the natural remedy, but VIF quantifies the problem | `variance_inflation_factor(X, i)` for each regressor |
| **No residual plots** | Normality and absence of patterns are Gauss-Markov prerequisites; statsmodels prints DW and JB but no visual diagnostic | `plot_regress_exog(result, 'exp')`, Q-Q plot of residuals |
| **Counterfactuals ≠ causal** | `skill_wage_premium` computes a conditional average treatment effect, not a causal effect; ability bias and selection confound the estimate | IV or propensity-score matching for causal identification |
| **n\_jobs=1 in cross\_val\_score** | Sequential CV is slow on large datasets; set to 1 here to avoid Windows paging-file OOM errors | Set `n_jobs=-1` on Linux/Mac or with adequate virtual memory |