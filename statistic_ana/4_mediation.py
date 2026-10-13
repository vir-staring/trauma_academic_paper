#!/usr/bin/env python3
"""Bootstrap mediation models: trauma -> brain -> behavior."""

import os
import sys

from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from _mediation_style import plot_simple_mediation


BRAIN_TABLE = Path("/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv")
ENV_TABLE = Path("/path/to/data/final/env_bhv_cov.csv")
SUBJECT_ID = "subject_id"

X_COLUMN = "childhood_trauma"
M_COLUMNS = [
    "Orbito-Affective_FA_grade_z",
]
Y_COLUMNS = [
    "ms_total_score_grade_z",
    "cr_total_score_grade_z"
]
COVARIATES = [
    "gender", 
    "f_ses", 
]

N_BOOT = 5000
RANDOM_SEED = 20260708
RESULT_TABLE = Path("/path/to/output/mediation/mediation_bootstrap_results.csv")
PLOT_DIR = RESULT_TABLE.parent / "simple_mediation_plots"


def ols(y, x):
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()


def make_model_data(data, x_column, m_column, y_column, covariates):
    model_data = data[[x_column, m_column, y_column] + covariates].copy()
    text_covariates = [c for c in covariates if model_data[c].dtype == "object"]
    model_data = pd.get_dummies(model_data, columns=text_covariates, drop_first=True)
    model_data = model_data.apply(pd.to_numeric, errors="coerce").dropna()
    covariate_columns = [c for c in model_data.columns if c not in [x_column, m_column, y_column]]
    return model_data, covariate_columns


def fit_mediation(model_data, x_column, m_column, y_column, covariate_columns):
    # a path: trauma -> brain, controlling covariates.
    a_model = ols(model_data[m_column], model_data[[x_column] + covariate_columns])

    # b and c' paths: behavior ~ trauma + brain + covariates.
    b_model = ols(model_data[y_column], model_data[[x_column, m_column] + covariate_columns])

    # c path: total association behavior ~ trauma + covariates.
    c_model = ols(model_data[y_column], model_data[[x_column] + covariate_columns])

    a = a_model.params[x_column]
    b = b_model.params[m_column]
    c_prime = b_model.params[x_column]
    c_total = c_model.params[x_column]
    return a_model, b_model, c_model, a, b, c_prime, c_total, a * b


def demo_frames(n=900):
    rng = np.random.default_rng(20260909)
    subject_id = [f"demo_{index:04d}" for index in range(n)]
    trauma, ses = rng.normal(1.45, .4, n), rng.normal(size=n)
    gender = rng.binomial(1, .47, n)
    mediator = -.02 * trauma + .10 * ses + rng.normal(size=n)
    brain = pd.DataFrame({SUBJECT_ID: subject_id, M_COLUMNS[0]: mediator})
    env = pd.DataFrame({SUBJECT_ID: subject_id, X_COLUMN: trauma,
                        "gender": gender, "f_ses": ses,
                        Y_COLUMNS[0]: -.30 * trauma + .02 * mediator + .15 * ses + rng.normal(size=n),
                        Y_COLUMNS[1]: -.34 * trauma + .02 * mediator + .15 * ses + rng.normal(size=n)})
    return brain, env


if "--demo" in sys.argv[1:]:
    N_BOOT = 400
    output_root = Path("/path/to/output/mediation_demo/simple")
    RESULT_TABLE = output_root / "mediation_bootstrap_results.csv"
    PLOT_DIR = output_root / "plots"
    brain, env = demo_frames()
else:
    brain, env = pd.read_csv(BRAIN_TABLE), pd.read_csv(ENV_TABLE)
data = brain.merge(env, on=SUBJECT_ID, how="inner", suffixes=("_brain", "_env"))

rng = np.random.default_rng(RANDOM_SEED)
rows = []

for m_column in M_COLUMNS:
    for y_column in Y_COLUMNS:
        model_data, covariate_columns = make_model_data(data, X_COLUMN, m_column, y_column, COVARIATES)
        a_model, b_model, c_model, a, b, c_prime, c_total, indirect = fit_mediation(
            model_data, X_COLUMN, m_column, y_column, covariate_columns
        )

        boot_indirect = []
        for _ in range(N_BOOT):
            sample_index = rng.integers(0, len(model_data), len(model_data))
            sample = model_data.iloc[sample_index]
            boot = fit_mediation(sample, X_COLUMN, m_column, y_column, covariate_columns)
            boot_indirect.append(boot[-1])

        boot_indirect = np.array(boot_indirect)
        ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
        boot_p = 2 * min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0))

        a_ci_low, a_ci_high = a_model.conf_int(alpha=0.05).loc[X_COLUMN]
        b_ci_low, b_ci_high = b_model.conf_int(alpha=0.05).loc[m_column]
        c_prime_ci_low, c_prime_ci_high = b_model.conf_int(alpha=0.05).loc[X_COLUMN]
        c_total_ci_low, c_total_ci_high = c_model.conf_int(alpha=0.05).loc[X_COLUMN]

        rows.append({
            "x": X_COLUMN,
            "mediator": m_column,
            "n_boot": N_BOOT,
            "a_beta": a,
            "a_se": a_model.bse[X_COLUMN],
            "a_ci95_low": a_ci_low,
            "a_ci95_high": a_ci_high,
            "a_t": a_model.tvalues[X_COLUMN],
            "a_p": a_model.pvalues[X_COLUMN],
            "b_beta": b,
            "b_se": b_model.bse[m_column],
            "b_ci95_low": b_ci_low,
            "b_ci95_high": b_ci_high,
            "b_t": b_model.tvalues[m_column],
            "b_p": b_model.pvalues[m_column],
            "c_prime_beta": c_prime,
            "c_prime_se": b_model.bse[X_COLUMN],
            "c_prime_ci95_low": c_prime_ci_low,
            "c_prime_ci95_high": c_prime_ci_high,
            "c_prime_t": b_model.tvalues[X_COLUMN],
            "c_prime_p": b_model.pvalues[X_COLUMN],
            "c_total_beta": c_total,
            "c_total_se": c_model.bse[X_COLUMN],
            "c_total_ci95_low": c_total_ci_low,
            "c_total_ci95_high": c_total_ci_high,
            "c_total_t": c_model.tvalues[X_COLUMN],
            "c_total_p": c_model.pvalues[X_COLUMN],
            "indirect_effect": indirect,
            "indirect_boot_mean": boot_indirect.mean(),
            "indirect_boot_se": boot_indirect.std(ddof=1),
            "indirect_ci95_low": ci_low,
            "indirect_ci95_high": ci_high,
            "indirect_boot_p": boot_p,
            "indirect_significant_95ci": ci_low > 0 or ci_high < 0,
            "a_model_r2": a_model.rsquared,
            "a_model_adj_r2": a_model.rsquared_adj,
            "b_model_r2": b_model.rsquared,
            "b_model_adj_r2": b_model.rsquared_adj,
            "c_model_r2": c_model.rsquared,
            "c_model_adj_r2": c_model.rsquared_adj,
        })

results = pd.DataFrame(rows)
results["indirect_boot_fdr_p"] = multipletests(results["indirect_boot_p"], method="fdr_bh")[1]
results["indirect_boot_fdr_reject_0.05"] = multipletests(results["indirect_boot_p"], method="fdr_bh")[0]

RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_TABLE, index=False, na_rep="")

plot_simple_mediation(results, Y_COLUMNS, PLOT_DIR / "simple_mediation_forest.svg")

print("Results:", RESULT_TABLE)
print("Plots:", PLOT_DIR)
