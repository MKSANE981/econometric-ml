"""
analysis.py — End-to-end pipeline: Mincer OLS → regularised regression → ML.

Data: Stack Overflow Developer Survey (download instructions in README).
Target: log(ConvertedCompYearly) — annual compensation in USD, log scale.

Usage:
    python analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.econometrics import (
    add_mincer_features,
    ols_mincer,
    compute_returns_to_education,
    describe_wage_distribution,
)
from src.ml_models import (
    build_ml_models,
    cross_validate_models,
    skill_wage_premium,
)
from src.visualization import (
    plot_wage_distribution,
    plot_mincer_coefficients,
    plot_experience_wage_profile,
    plot_model_comparison,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SURVEY_PATH = Path("data/survey_results_public.csv")


def load_stackoverflow_survey(path: Path) -> pd.DataFrame:
    """
    Load and clean the Stack Overflow Developer Survey.

    Key columns used:
    - ConvertedCompYearly: annual salary in USD (target)
    - YearsCodePro: years of professional coding experience
    - EdLevel: highest education level
    - Country: country of respondent
    - LanguageHaveWorkedWith: languages used
    """
    df = pd.read_csv(path, low_memory=False)

    # Filter to full-time employed developers with valid salary
    df = df[
        (df["Employment"].str.contains("full-time", case=False, na=False))
        & (df["ConvertedCompYearly"].notna())
        & (df["ConvertedCompYearly"] > 1000)
        & (df["ConvertedCompYearly"] < 500_000)
    ].copy()

    df["log_salary"] = np.log(df["ConvertedCompYearly"])
    df["exp"] = pd.to_numeric(df["YearsCodePro"], errors="coerce").fillna(df["YearsCodePro"].median())
    df["exp_sq"] = df["exp"] ** 2

    # Education ordinal encoding
    edu_map = {
        "Primary/elementary school": 0,
        "Secondary school": 1,
        "Some college/university study without earning a degree": 2,
        "Associate degree": 3,
        "Bachelor's degree": 4,
        "Master's degree": 5,
        "Professional degree": 6,
        "Other doctoral degree": 7,
    }
    df["edu"] = df["EdLevel"].map(edu_map).fillna(3)

    # Top language dummies
    top_langs = ["Python", "JavaScript", "SQL", "TypeScript", "Rust", "Go", "Java", "C++"]
    for lang in top_langs:
        df[f"lang_{lang.lower()}"] = df["LanguageHaveWorkedWith"].str.contains(lang, na=False).astype(int)

    return df.dropna(subset=["log_salary", "exp", "edu"])


def run_econometric_analysis(df: pd.DataFrame) -> None:
    print("\n=== 1. WAGE DISTRIBUTION ===")
    stats = describe_wage_distribution(df["ConvertedCompYearly"])
    print(stats)
    plot_wage_distribution(df["ConvertedCompYearly"], df["log_salary"])

    print("\n=== 2. MINCER OLS (HC3 robust SEs) ===")
    formula = "log_salary ~ edu + exp + exp_sq"
    result = ols_mincer(df, formula, robust_se=True)
    print(result.summary())
    plot_mincer_coefficients(result)
    plot_experience_wage_profile(result)

    returns = compute_returns_to_education(result)
    print(f"\n  Return to education (approx): {returns['approx_pct_return']:.1f}% per level")
    print(f"  Return to education (exact):  {returns['exact_pct_return']:.1f}% per level")

    result.save(RESULTS_DIR / "ols_mincer.pkl")


def run_ml_analysis(df: pd.DataFrame) -> None:
    print("\n=== 3. ML MODELS ===")
    feature_cols = (
        ["edu", "exp", "exp_sq"]
        + [c for c in df.columns if c.startswith("lang_")]
    )
    X = df[feature_cols].fillna(0)
    y = df["log_salary"]

    models = build_ml_models()
    results = cross_validate_models(models, X, y)
    print(results[["model", "rmse_log", "r2"]].to_string(index=False))
    results.to_csv(RESULTS_DIR / "ml_results.csv", index=False)
    plot_model_comparison(results)

    print("\n=== 4. SKILL WAGE PREMIA (via ML counterfactuals) ===")
    best_model_name = results.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X, y)

    for lang in ["lang_python", "lang_rust", "lang_sql"]:
        if lang in X.columns:
            skill_wage_premium(best_model, X, skill_col=lang)


if __name__ == "__main__":
    if not SURVEY_PATH.exists():
        print("Download the Stack Overflow Developer Survey from:")
        print("  https://survey.stackoverflow.co/")
        print(f"and place survey_results_public.csv in data/")
        raise SystemExit(1)

    df = load_stackoverflow_survey(SURVEY_PATH)
    print(f"Loaded {len(df):,} developer records")

    run_econometric_analysis(df)
    run_ml_analysis(df)

    print("\nDone. Results saved to results/")