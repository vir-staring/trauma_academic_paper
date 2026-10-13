#!/usr/bin/env python3
"""Partial correlation between brain metrics and behavior/environment variables."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from _partial_correlation_style import plot_scatter, plot_summary


BRAIN_TABLE = Path("/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv")
ENV_TABLE = Path("/path/to/data/final/env_bhv_cov.csv")
SUBJECT_ID = "subject_id"

X_COLUMNS = [
    "childhood_trauma",

]
Y_COLUMNS = [

    # "Visual1_FA_grade_z",
    # "Visual2_FA_grade_z",
    # "Auditory_FA_grade_z",
    # "Somatomotor_FA_grade_z",
    "Orbito-Affective_FA_grade_z",
    # "Ventral-Multimodal_FA_grade_z",
    # "Posterior-Multimodal_FA_grade_z",
    # "Cingulo-Opercular_FA_grade_z",
    # "Default_FA_grade_z",
    # "Language_FA_grade_z",
    # "Frontoparietal_FA_grade_z",
    # "Dorsal-Attention_FA_grade_z"
    
]
COVARIATES = [
    "gender", 
    "f_ses", 
    # "age_months"
    ]

OUT_DIR = "/path/to/output/cor/env-brain/"
RESULT_TABLE = Path(f"{OUT_DIR}/partial_correlation_results.csv")
HEATMAP = Path(f"{OUT_DIR}/partial_correlation_heatmap.svg")
SCATTER_DIR = Path(f"{OUT_DIR}/partial_correlation_scatter")


def residualize(value, covariates):
    covariates = sm.add_constant(covariates, has_constant="add")
    return sm.OLS(value, covariates).fit().resid


def safe_name(text):
    for old in ["/", "\\", " ", ":", "*", "?", "\"", "<", ">", "|"]:
        text = text.replace(old, "_")
    return text


def demo_frames(n=1200):
    rng = np.random.default_rng(20260909)
    subject_id = [f"demo_{index:04d}" for index in range(n)]
    trauma = rng.normal(1.45, .40, n)
    gender = rng.binomial(1, .47, n)
    ses = rng.normal(size=n)
    env = pd.DataFrame({SUBJECT_ID: subject_id, X_COLUMNS[0]: trauma,
                        COVARIATES[0]: gender, COVARIATES[1]: ses})
    brain = pd.DataFrame({SUBJECT_ID: subject_id,
                          Y_COLUMNS[0]: .08 * gender + .12 * ses + rng.normal(size=n)})
    return brain, env


if "--demo" in sys.argv[1:]:
    output_root = Path("/path/to/output/partial_correlation_demo/env-brain")
    RESULT_TABLE = output_root / "partial_correlation_results.csv"
    HEATMAP = output_root / "partial_correlation_heatmap.svg"
    SCATTER_DIR = output_root / "partial_correlation_scatter"
    brain, env = demo_frames()
else:
    brain, env = pd.read_csv(BRAIN_TABLE), pd.read_csv(ENV_TABLE)
data = brain.merge(env, on=SUBJECT_ID, how="inner", suffixes=("_brain", "_env"))

need_columns = [SUBJECT_ID] + X_COLUMNS + Y_COLUMNS + COVARIATES
data = data[need_columns].copy()
data = pd.get_dummies(data, columns=[c for c in COVARIATES if data[c].dtype == "object"], drop_first=True)

covariate_columns = [c for c in data.columns if c not in [SUBJECT_ID] + X_COLUMNS + Y_COLUMNS]
rows = []
scatter_data = {}

for x_column in X_COLUMNS:
    for y_column in Y_COLUMNS:
        model_data = data[[x_column, y_column] + covariate_columns].apply(pd.to_numeric, errors="coerce").dropna()
        x_resid = residualize(model_data[x_column], model_data[covariate_columns])
        y_resid = residualize(model_data[y_column], model_data[covariate_columns])
        r, p = stats.pearsonr(x_resid, y_resid)
        df = len(model_data) - len(covariate_columns) - 2
        t_value = r * np.sqrt(df / (1 - r * r))

        full_x = sm.add_constant(model_data[[x_column] + covariate_columns], has_constant="add")
        model = sm.OLS(model_data[y_column], full_x).fit()
        ci_low, ci_high = model.conf_int().loc[x_column]

        rows.append({
            "x": x_column,
            "y": y_column,
            "covariates": "; ".join(covariate_columns),
            "n": len(model_data),
            "df": df,
            "partial_r": r,
            "partial_t": t_value,
            "partial_p": p,
            "ols_beta_x": model.params[x_column],
            "ols_se_x": model.bse[x_column],
            "ols_t_x": model.tvalues[x_column],
            "ols_p_x": model.pvalues[x_column],
            "ols_ci95_low_x": ci_low,
            "ols_ci95_high_x": ci_high,
            "model_r2": model.rsquared,
            "model_adj_r2": model.rsquared_adj,
            "model_f": model.fvalue,
            "model_f_p": model.f_pvalue,
        })
        scatter_data[(x_column, y_column)] = (x_resid, y_resid, r, p, len(model_data))

results = pd.DataFrame(rows)
results["fdr_p"] = multipletests(results["partial_p"], method="fdr_bh")[1]
results["fdr_reject_0.05"] = multipletests(results["partial_p"], method="fdr_bh")[0]

RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_TABLE, index=False, na_rep="")

plot_summary(results, HEATMAP, "Trauma and orbito-affective network FA")

# Residual scatter plots for every X-Y pair.
SCATTER_DIR.mkdir(parents=True, exist_ok=True)
for (x_column, y_column), (x_resid, y_resid, r, p, n) in scatter_data.items():
    q = results.loc[(results["x"] == x_column) & (results["y"] == y_column), "fdr_p"].iat[0]
    plot_scatter(x_resid, y_resid, x_column, y_column, r, q, n,
                 SCATTER_DIR / (safe_name(x_column + "__" + y_column) + ".svg"))

print("Results:", RESULT_TABLE)
print("Summary plot:", HEATMAP)
print("Scatter plots:", SCATTER_DIR)
