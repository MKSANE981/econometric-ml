"""
Generate synthetic survey data matching the Stack Overflow Developer Survey structure.

The real survey is available at https://survey.stackoverflow.co/ (~100 MB).
This script generates a realistic synthetic version (2 000 rows) for demo purposes,
preserving the expected column names and distributions.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

ED_LEVELS = [
    "Primary/elementary school",
    "Secondary school",
    "Some college/university study without earning a degree",
    "Associate degree",
    "Bachelor's degree",
    "Master's degree",
    "Professional degree",
    "Other doctoral degree",
]
ED_WEIGHTS = [0.01, 0.03, 0.08, 0.05, 0.45, 0.28, 0.05, 0.05]

LANGUAGES = ["Python", "JavaScript", "SQL", "TypeScript", "Rust", "Go", "Java", "C++"]
LANG_PROBS = [0.65, 0.60, 0.55, 0.40, 0.12, 0.18, 0.35, 0.22]

edu_idx = rng.choice(len(ED_LEVELS), size=N, p=ED_WEIGHTS)
edu_num = edu_idx.astype(float)

exp = np.clip(rng.normal(loc=7, scale=5, size=N), 0, 35).round(1)

# Mincer-like salary: log(w) = 10.5 + 0.08*edu + 0.05*exp - 0.001*exp^2 + noise
log_salary = (
    10.5
    + 0.08 * edu_num
    + 0.05 * exp
    - 0.001 * exp ** 2
    + rng.normal(0, 0.35, N)
)
salary = np.exp(log_salary).clip(10_000, 490_000).round()

# Build language lists
lang_lists = []
for _ in range(N):
    langs = [l for l, p in zip(LANGUAGES, LANG_PROBS) if rng.random() < p]
    if not langs:
        langs = [rng.choice(LANGUAGES)]
    lang_lists.append(";".join(langs))

df = pd.DataFrame({
    "Employment": rng.choice(
        ["Employed, full-time", "Employed, part-time", "Student"],
        size=N, p=[0.80, 0.10, 0.10],
    ),
    "ConvertedCompYearly": salary,
    "YearsCodePro": exp.astype(str),
    "EdLevel": [ED_LEVELS[i] for i in edu_idx],
    "Country": rng.choice(["United States", "Germany", "France", "India", "United Kingdom"], size=N),
    "LanguageHaveWorkedWith": lang_lists,
})

df.to_csv("data/survey_results_public.csv", index=False)
print(f"Saved {len(df)} rows -> data/survey_results_public.csv")
print("Columns:", list(df.columns))
