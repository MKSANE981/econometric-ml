"""
visualization.py — Plots for econometric and ML wage analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def plot_wage_distribution(y_raw: pd.Series, y_log: pd.Series) -> None:
    """
    Side-by-side: raw wage distribution vs log(wage).
    Demonstrates why the log transformation is appropriate (right-skew → near-normal).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(y_raw, bins=60, color="#2196F3", alpha=0.8, edgecolor="white")
    ax1.set_title("Salary distribution (raw)", fontsize=12)
    ax1.set_xlabel("Annual salary (USD)")

    ax2.hist(y_log, bins=60, color="#FF5722", alpha=0.8, edgecolor="white")
    ax2.set_title("log(Salary) distribution", fontsize=12)
    ax2.set_xlabel("log(Annual salary)")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "wage_distribution.png", dpi=150)
    plt.close(fig)


def plot_mincer_coefficients(result, title: str = "OLS Mincer Equation") -> None:
    """
    Horizontal coefficient plot with 95% confidence intervals.
    Standard way to visualise regression results — more readable than a table.
    """
    coefs = result.params.drop("Intercept", errors="ignore")
    cis = result.conf_int().drop("Intercept", errors="ignore")

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(coefs) + 2))
    y_pos = np.arange(len(coefs))

    ax.barh(y_pos, coefs, xerr=[coefs - cis[0], cis[1] - coefs],
            color=["#2196F3" if v > 0 else "#F44336" for v in coefs],
            alpha=0.8, capsize=4, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index, fontsize=10)
    ax.set_xlabel("Coefficient (log-wage scale)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "mincer_coefficients.png", dpi=150)
    plt.close(fig)


def plot_experience_wage_profile(result, exp_range: tuple = (0, 35)) -> None:
    """
    Plot the non-linear experience-wage profile implied by the Mincer model.

    From β₂ and β₃:
        Predicted Δln(w) = β₂·Exp + β₃·Exp²

    Peak experience: Exp* = -β₂ / (2·β₃)
    """
    exp = np.linspace(*exp_range, 200)
    params = result.params

    b2 = params.get("exp", params.get("YearsCodePro", 0))
    b3 = params.get("exp_sq", params.get("exp_squared", 0))

    predicted = b2 * exp + b3 * exp ** 2
    peak_exp = -b2 / (2 * b3) if b3 != 0 else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(exp, predicted, color="#2196F3", linewidth=2)
    if peak_exp and exp_range[0] <= peak_exp <= exp_range[1]:
        ax.axvline(peak_exp, color="#FF5722", linestyle="--",
                   label=f"Peak experience ≈ {peak_exp:.1f} years")
    ax.set_xlabel("Years of experience", fontsize=12)
    ax.set_ylabel("Δ log(wage)", fontsize=12)
    ax.set_title("Non-linear experience–wage profile (Mincer)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "experience_wage_profile.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame) -> None:
    """Bar chart comparing ML models and OLS on RMSE (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(results) + 1.5))
    colors = ["#4CAF50" if r < results["rmse_log"].median() else "#2196F3"
              for r in results["rmse_log"]]

    ax.barh(results["model"], results["rmse_log"],
            xerr=results["rmse_std"], color=colors, alpha=0.85,
            capsize=4, edgecolor="white")
    ax.set_xlabel("RMSE (log-salary scale)", fontsize=12)
    ax.set_title("Econometric vs ML — Wage Prediction (5-fold CV)", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, model_name: str) -> None:
    """SHAP beeswarm plot — shows direction and magnitude of each feature's effect."""
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="dot")
        plt.title(f"SHAP Feature Impact — {model_name}")
        plt.tight_layout()
        safe = model_name.lower().replace(" ", "_")
        plt.savefig(RESULTS_DIR / f"shap_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass