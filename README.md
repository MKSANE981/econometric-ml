# Econometric-ML — Causal Inference vs. Predictive Modelling of Wages

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

## Quickstart

```bash
pip install -r requirements.txt
python analysis.py
```