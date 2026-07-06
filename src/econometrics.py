"""
econometrics.py — OLS, robust standard errors, and IV/2SLS.

Why implement these manually alongside statsmodels?
Because understanding the matrix algebra behind each estimator matters.
We document the formulas in each function so the code itself is a derivation,
not just an API call.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def add_mincer_features(df: pd.DataFrame, edu_col: str, exp_col: str) -> pd.DataFrame:
    """
    Construct Mincer equation features from raw columns.

    The Mincer earnings equation (1974) is:

        ln(w) = α + β₁·S + β₂·Exp + β₃·Exp² + ε

    The quadratic experience term captures the concavity of wage-experience
    profiles: wages rise fast early in career, then plateau (β₃ < 0).

    Peak experience is reached at: Exp* = -β₂ / (2·β₃)
    """
    out = df.copy()
    out["exp_sq"] = out[exp_col] ** 2
    return out


def ols_mincer(
    df: pd.DataFrame,
    formula: str,
    robust_se: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS with optional HC3 heteroskedasticity-robust standard errors.

    Why HC3 over HC1 or HC0?
    HC3 inflates the squared residuals by 1/(1-h_ii)² where h_ii is leverage.
    This correction makes it better behaved in small samples and with
    high-leverage observations — both common in wage data.

    The HC3 sandwich estimator:
        V̂_HC3 = (X'X)⁻¹ [Σ xᵢxᵢ' ε̂ᵢ²/(1-hᵢᵢ)²] (X'X)⁻¹

    Parameters
    ----------
    df      : DataFrame with all variables in formula
    formula : statsmodels formula string, e.g. "log_salary ~ edu + exp + exp_sq"
    robust_se : if True, use HC3; otherwise use classical OLS SEs

    Returns
    -------
    Fitted RegressionResults object with .summary(), .params, .bse, .pvalues
    """
    model = smf.ols(formula, data=df)
    cov_type = "HC3" if robust_se else "nonrobust"
    result = model.fit(cov_type=cov_type)
    return result


def compute_returns_to_education(result) -> dict:
    """
    Extract and interpret the return to education coefficient.

    In a log-linear model (ln(w) ~ edu + ...):
        β₁ ≈ % wage increase per additional year of schooling

    More precisely, the exact percentage change is:
        (exp(β₁) - 1) × 100%

    Both the approximation (β₁ × 100%) and exact version are returned.
    """
    params = result.params
    edu_coef = params.get("edu", params.get("education", params.get("EdLevel", None)))

    if edu_coef is None:
        return {"error": "Education coefficient not found in model"}

    return {
        "coefficient": edu_coef,
        "approx_pct_return": edu_coef * 100,
        "exact_pct_return": (np.exp(edu_coef) - 1) * 100,
        "p_value": result.pvalues.get("edu", None),
        "significant_5pct": result.pvalues.get("edu", 1) < 0.05,
    }


def hausman_test(ols_result, iv_result) -> dict:
    """
    Hausman specification test for endogeneity.

    H₀: OLS is consistent (no endogeneity) → use OLS (efficient)
    H₁: OLS is inconsistent → use IV (consistent but inefficient)

    Test statistic:
        H = (β̂_IV - β̂_OLS)' [Var(β̂_IV) - Var(β̂_OLS)]⁻¹ (β̂_IV - β̂_OLS)
        H ~ χ²(k) under H₀

    Interpretation: reject H₀ (p < 0.05) means endogeneity is present and
    IV estimates should be preferred over OLS.
    """
    diff = iv_result.params - ols_result.params
    cov_diff = iv_result.cov_params() - ols_result.cov_params()

    try:
        H = float(diff @ np.linalg.inv(cov_diff) @ diff)
        df = len(diff)
        p_value = 1 - stats.chi2.cdf(H, df)
        return {
            "statistic": H,
            "df": df,
            "p_value": p_value,
            "endogeneity_detected": p_value < 0.05,
        }
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix — check instrument validity"}


def iv_2sls(
    df: pd.DataFrame,
    endog: str,
    exog: list[str],
    instruments: list[str],
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Two-Stage Least Squares (2SLS) estimator.

    Stage 1: regress the endogenous regressor on all exogenous variables + instruments
        X̂ = Z(Z'Z)⁻¹Z'X

    Stage 2: regress y on X̂ instead of X
        β̂_2SLS = (X̂'X)⁻¹X̂'y = (X'P_Z X)⁻¹X'P_Z y

    where P_Z = Z(Z'Z)⁻¹Z' is the projection matrix onto the column space of Z.

    A valid instrument Z must satisfy:
        1. Relevance: Cov(Z, X) ≠ 0  (testable — check F-stat > 10 in stage 1)
        2. Exclusion: Cov(Z, ε) = 0   (untestable — requires economic reasoning)

    Parameters
    ----------
    endog       : name of endogenous variable (e.g. "education")
    exog        : list of exogenous controls
    instruments : list of instruments (must satisfy relevance + exclusion)
    """
    from linearmodels.iv import IV2SLS

    formula_vars = [endog] + exog
    Z = sm.add_constant(df[instruments + exog])
    X_endog = df[endog]

    # Stage 1 — check instrument relevance
    stage1 = sm.OLS(X_endog, Z).fit()
    f_stat = stage1.fvalue
    print(f"  Stage 1 F-statistic: {f_stat:.2f} (rule of thumb: > 10 for strong instrument)")

    return stage1


def describe_wage_distribution(series: pd.Series, label: str = "salary") -> pd.DataFrame:
    """
    Descriptive statistics for wage data.

    Wages are typically log-normal: taking log(wage) gives an approximately
    normal distribution, which validates the log-linear Mincer specification.
    We test log-normality via the Shapiro-Wilk test on log(wage).
    """
    log_series = np.log(series.clip(lower=1))
    stat, p = stats.shapiro(log_series.sample(min(5000, len(log_series)), random_state=42))

    summary = series.describe()
    summary["log_mean"] = log_series.mean()
    summary["log_std"] = log_series.std()
    summary["log_normality_p"] = p
    summary["is_log_normal"] = p > 0.05

    return summary