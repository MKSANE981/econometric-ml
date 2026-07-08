"""
ml_models.py — ML models for wage prediction with CV and SHAP interpretation.

The key question we answer here is not just "which model predicts best?"
but "what does each model learn that OLS misses?" — non-linearities,
interaction effects, and threshold effects in skill premia.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def build_ml_models() -> dict:
    """
    Return all ML models for wage prediction.

    We include regularised linear models alongside tree-based models to show
    the bias-variance spectrum:
    - Ridge / Lasso: linear but regularised — reveal which skills are
      selected by L1 shrinkage (Lasso) vs. which are merely shrunk (Ridge)
    - Random Forest: non-linear, robust to outliers, gives feature importance
    - GBM / XGBoost: best predictive performance, captures interactions
      between skills and experience

    Note on the target: we model log(salary), so predictions must be
    exponentiated to get dollar values. RMSE on log scale is more meaningful
    than on the raw scale because wages are right-skewed.
    """
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=5000)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=10,
                min_samples_leaf=5, random_state=42, n_jobs=1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42,
            )),
        ]),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5, random_state=42, n_jobs=1, verbosity=0,
            )),
        ])

    return models




def cross_validate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run 5-fold CV for all models, return sorted comparison DataFrame.

    Metrics on log(salary) scale:
    - RMSE: root mean squared error (interpretable as ± log-salary)
    - R²: proportion of variance explained (1 = perfect)

    To interpret RMSE on log scale: RMSE ≈ 0.3 means predictions are off
    by roughly ±30% in salary (since log(1.3) ≈ 0.26).
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for name, model in models.items():
        rmse_scores = -cross_val_score(
            model, X, y, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)

        results.append({
            "model": name,
            "rmse_log": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
            "r2": r2_scores.mean(),
            "r2_std": r2_scores.std(),
        })

    df = pd.DataFrame(results).sort_values("rmse_log").reset_index(drop=True)
    return df


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 500) -> tuple:
    """
    Compute SHAP values for a fitted tree-based model.

    SHAP (SHapley Additive exPlanations) decomposes each prediction into
    additive contributions from each feature — grounded in game theory:

        f(x) = E[f(X)] + Σⱼ φⱼ(x)

    where φⱼ is the Shapley value for feature j:

        φⱼ = Σ_{S ⊆ F\{j}} [|S|!(|F|-|S|-1)!/|F|!] × [f(S∪{j}) - f(S)]

    This is the only method satisfying all four desirable properties:
    efficiency, symmetry, dummy, and additivity.

    Unlike impurity importance, SHAP values are not biased toward
    high-cardinality features and capture interaction effects.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed. Run: pip install shap")
        return None, None

    sample = X.sample(min(sample_size, len(X)), random_state=42)

    inner = model.named_steps.get("model") or list(model.named_steps.values())[-1]

    if hasattr(inner, "feature_importances_"):
        explainer = shap.TreeExplainer(inner)
    else:
        explainer = shap.LinearExplainer(inner, sample)

    shap_values = explainer.shap_values(sample)
    return shap_values, sample


def skill_wage_premium(
    model,
    X: pd.DataFrame,
    skill_col: str,
    baseline: int = 0,
    treatment: int = 1,
) -> float:
    """
    Estimate the causal wage premium for having a skill using the ML model.

    This is a simple counterfactual: we predict wages for all individuals
    as if they have the skill (treatment=1) vs. don't (baseline=0),
    then take the average difference in log-wages.

    Note: this is NOT causal without additional assumptions — it captures
    the conditional association between skill and wages, controlling for
    all other features in X. For a causal interpretation, we would need
    an exogenous source of variation in skill acquisition.

    Returns the average treatment effect in log-wage units.
    (Multiply by 100 for approximate percentage wage premium.)
    """
    X_treat = X.copy()
    X_base = X.copy()
    X_treat[skill_col] = treatment
    X_base[skill_col] = baseline

    pred_treat = model.predict(X_treat)
    pred_base = model.predict(X_base)
    ate = (pred_treat - pred_base).mean()

    print(f"  Skill premium for {skill_col}: {ate*100:.1f}% (log-wage ATE)")
    return ate