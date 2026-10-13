#!/usr/bin/env python3
"""Age-moderated mediation: trauma -> brain -> behavior."""

import os
import sys

from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from _mediation_style import plot_age_moderated_mediation


BRAIN_TABLE = Path("/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv")
ENV_TABLE = Path("/path/to/data/final/env_bhv_cov.csv")
SUBJECT_ID = "subject_id"

MODERATOR = "age_month"

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
AGE_POINTS = ["mean_minus_sd", "mean", "mean_plus_sd"]
AGE_CONTRASTS = [
    ("mean_minus_sd", "mean"),
    ("mean_minus_sd", "mean_plus_sd"),
    ("mean", "mean_plus_sd"),
]

N_BOOT = 5000
RANDOM_SEED = 20260708
RESULT_TABLE = Path("/path/to/output/mediation/age_moderated_mediation_bootstrap_results.csv")
CONTRAST_TABLE = Path(
    "/path/to/output/mediation/age_moderated_mediation_indirect_age_contrasts.csv"
)
PLOT_DIR = RESULT_TABLE.parent / "age_moderated_mediation_plots"

def ols(y, x):
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()


def make_model_data(data, x_column, m_column, y_column, moderator, covariates):
    columns = [x_column, m_column, y_column, moderator] + covariates
    model_data = data[columns].copy()
    text_covariates = [c for c in covariates if model_data[c].dtype == "object"]
    model_data = pd.get_dummies(model_data, columns=text_covariates, drop_first=True)
    model_data = model_data.apply(pd.to_numeric, errors="coerce").dropna()
    model_data["age_c"] = model_data[moderator] - model_data[moderator].mean()
    model_data["x_age"] = model_data[x_column] * model_data["age_c"]
    model_data["m_age"] = model_data[m_column] * model_data["age_c"]
    covariate_columns = [c for c in model_data.columns if c not in [
        x_column, m_column, y_column, moderator, "age_c", "x_age", "m_age"
    ]]
    return model_data, covariate_columns


def fit_model(model_data, x_column, m_column, y_column, covariate_columns):
    # a path is moderated by age: M ~ X + age + X*age + covariates.
    a_x = [x_column, "age_c", "x_age"] + covariate_columns
    a_model = ols(model_data[m_column], model_data[a_x])

    # b path and direct c' path are moderated by age.
    b_x = [x_column, m_column, "age_c", "x_age", "m_age"] + covariate_columns
    b_model = ols(model_data[y_column], model_data[b_x])

    return a_model, b_model


def conditional_indirect(a_model, b_model, age_c, x_column, m_column):
    a_path = a_model.params[x_column] + a_model.params["x_age"] * age_c
    b_path = b_model.params[m_column] + b_model.params["m_age"] * age_c
    return a_path, b_path, a_path * b_path


def linear_combination(model, term_weights):
    contrast = np.zeros(len(model.params), dtype=float)
    for term, weight in term_weights.items():
        contrast[model.params.index.get_loc(term)] = weight

    test = model.t_test(contrast)
    ci_low, ci_high = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)[:2]
    return {
        "beta": float(np.asarray(test.effect).reshape(-1)[0]),
        "se": float(np.asarray(test.sd).reshape(-1)[0]),
        "t": float(np.asarray(test.tvalue).reshape(-1)[0]),
        "p": float(np.asarray(test.pvalue).reshape(-1)[0]),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def demo_frames(n=900):
    rng = np.random.default_rng(20260909)
    subject_id = [f"demo_{index:04d}" for index in range(n)]
    trauma, ses = rng.normal(1.45, .4, n), rng.normal(size=n)
    gender, age = rng.binomial(1, .47, n), rng.uniform(108, 216, n)
    age_c = (age - age.mean()) / age.std()
    mediator = (-.02 * trauma + .01 * trauma * age_c + .10 * ses
                + rng.normal(size=n))
    brain = pd.DataFrame({SUBJECT_ID: subject_id, M_COLUMNS[0]: mediator})
    env = pd.DataFrame({SUBJECT_ID: subject_id, X_COLUMN: trauma,
                        MODERATOR: age, "gender": gender, "f_ses": ses,
                        Y_COLUMNS[0]: -.30 * trauma + .02 * mediator + .15 * ses + rng.normal(size=n),
                        Y_COLUMNS[1]: -.34 * trauma + .02 * mediator + .15 * ses + rng.normal(size=n)})
    return brain, env


if "--demo" in sys.argv[1:]:
    N_BOOT = 400
    output_root = Path("/path/to/output/mediation_demo/age_moderated")
    RESULT_TABLE = output_root / "age_moderated_mediation_bootstrap_results.csv"
    CONTRAST_TABLE = output_root / "age_moderated_mediation_indirect_age_contrasts.csv"
    PLOT_DIR = output_root / "plots"
    brain, env = demo_frames()
else:
    brain, env = pd.read_csv(BRAIN_TABLE), pd.read_csv(ENV_TABLE)
data = brain.merge(env, on=SUBJECT_ID, how="inner", suffixes=("_brain", "_env"))

rng = np.random.default_rng(RANDOM_SEED)
rows = []
contrast_rows = []

for m_column in M_COLUMNS:
    for y_column in Y_COLUMNS:
        model_data, covariate_columns = make_model_data(
            data, X_COLUMN, m_column, y_column, MODERATOR, COVARIATES
        )
        age_mean = model_data[MODERATOR].mean()
        age_sd = model_data[MODERATOR].std()
        age_values = {
            "mean_minus_sd": age_mean - age_sd,
            "mean": age_mean,
            "mean_plus_sd": age_mean + age_sd,
        }

        a_model, b_model = fit_model(model_data, X_COLUMN, m_column, y_column, covariate_columns)

        a_x_age_ci_low, a_x_age_ci_high = a_model.conf_int(alpha=0.05).loc["x_age"]
        b_m_age_ci_low, b_m_age_ci_high = b_model.conf_int(alpha=0.05).loc["m_age"]
        c_prime_x_age_ci_low, c_prime_x_age_ci_high = b_model.conf_int(alpha=0.05).loc["x_age"]

        boot_values = {label: [] for label in AGE_POINTS}
        for _ in range(N_BOOT):
            sample_index = rng.integers(0, len(model_data), len(model_data))
            sample = model_data.iloc[sample_index]
            boot_a, boot_b = fit_model(sample, X_COLUMN, m_column, y_column, covariate_columns)

            for label in AGE_POINTS:
                age_c = age_values[label] - age_mean
                boot_values[label].append(conditional_indirect(boot_a, boot_b, age_c, X_COLUMN, m_column)[2])

        conditional_indirect_estimates = {}
        for label in AGE_POINTS:
            age_raw = age_values[label]
            age_c = age_raw - age_mean
            a_path, b_path, indirect = conditional_indirect(a_model, b_model, age_c, X_COLUMN, m_column)
            conditional_indirect_estimates[label] = indirect
            boot_indirect = np.array(boot_values[label])
            ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
            boot_p = 2 * min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0))
            direct = linear_combination(
                b_model,
                {X_COLUMN: 1.0, "x_age": age_c},
            )

            rows.append({
                "x": X_COLUMN,
                "mediator": m_column,
                "y": y_column,
                "moderator": MODERATOR,
                "age_point": label,
                "age_months": age_raw,
                "age_centered": age_c,
                "covariates": "; ".join(covariate_columns),
                "n": len(model_data),
                "n_boot": N_BOOT,
                "a_x_beta": a_model.params[X_COLUMN],
                "a_x_p": a_model.pvalues[X_COLUMN],
                "a_age_beta": a_model.params["age_c"],
                "a_age_p": a_model.pvalues["age_c"],
                "a_x_age_beta": a_model.params["x_age"],
                "a_x_age_se": a_model.bse["x_age"],
                "a_x_age_ci95_low": a_x_age_ci_low,
                "a_x_age_ci95_high": a_x_age_ci_high,
                "a_x_age_t": a_model.tvalues["x_age"],
                "a_x_age_p": a_model.pvalues["x_age"],
                "b_m_beta": b_model.params[m_column],
                "b_m_p": b_model.pvalues[m_column],
                "b_age_beta": b_model.params["age_c"],
                "b_age_p": b_model.pvalues["age_c"],
                "b_m_age_beta": b_model.params["m_age"],
                "b_m_age_se": b_model.bse["m_age"],
                "b_m_age_ci95_low": b_m_age_ci_low,
                "b_m_age_ci95_high": b_m_age_ci_high,
                "b_m_age_t": b_model.tvalues["m_age"],
                "b_m_age_p": b_model.pvalues["m_age"],
                "c_prime_x_beta": b_model.params[X_COLUMN],
                "c_prime_x_p": b_model.pvalues[X_COLUMN],
                "c_prime_x_age_beta": b_model.params["x_age"],
                "c_prime_x_age_se": b_model.bse["x_age"],
                "c_prime_x_age_ci95_low": c_prime_x_age_ci_low,
                "c_prime_x_age_ci95_high": c_prime_x_age_ci_high,
                "c_prime_x_age_t": b_model.tvalues["x_age"],
                "c_prime_x_age_p": b_model.pvalues["x_age"],
                "conditional_a_path": a_path,
                "conditional_b_path": b_path,
                "conditional_direct": direct["beta"],
                "conditional_direct_se": direct["se"],
                "conditional_direct_ci95_low": direct["ci95_low"],
                "conditional_direct_ci95_high": direct["ci95_high"],
                "conditional_direct_t": direct["t"],
                "conditional_direct_p": direct["p"],
                "conditional_indirect": indirect,
                "conditional_indirect_boot_mean": boot_indirect.mean(),
                "conditional_indirect_boot_se": boot_indirect.std(ddof=1),
                "conditional_indirect_ci95_low": ci_low,
                "conditional_indirect_ci95_high": ci_high,
                "conditional_indirect_boot_p": boot_p,
                "conditional_indirect_significant_95ci": ci_low > 0 or ci_high < 0,
                "a_model_r2": a_model.rsquared,
                "a_model_adj_r2": a_model.rsquared_adj,
                "b_model_r2": b_model.rsquared,
                "b_model_adj_r2": b_model.rsquared_adj,
            })

        # When age moderates both a and b paths, the indirect effect is nonlinear
        # in age. Test moderation using bootstrap differences between selected ages.
        for point_1, point_2 in AGE_CONTRASTS:
            boot_difference = (
                np.asarray(boot_values[point_2]) - np.asarray(boot_values[point_1])
            )
            difference_ci_low, difference_ci_high = np.percentile(
                boot_difference, [2.5, 97.5]
            )
            difference_boot_p = 2 * min(
                np.mean(boot_difference <= 0),
                np.mean(boot_difference >= 0),
            )
            contrast_rows.append({
                "x": X_COLUMN,
                "mediator": m_column,
                "y": y_column,
                "moderator": MODERATOR,
                "contrast": f"{point_2}_minus_{point_1}",
                "age_point_1": point_1,
                "age_point_2": point_2,
                "age_1_months": age_values[point_1],
                "age_2_months": age_values[point_2],
                "age_1_centered": age_values[point_1] - age_mean,
                "age_2_centered": age_values[point_2] - age_mean,
                "covariates": "; ".join(covariate_columns),
                "n": len(model_data),
                "n_boot": N_BOOT,
                "conditional_indirect_difference": (
                    conditional_indirect_estimates[point_2]
                    - conditional_indirect_estimates[point_1]
                ),
                "difference_boot_mean": boot_difference.mean(),
                "difference_boot_se": boot_difference.std(ddof=1),
                "difference_ci95_low": difference_ci_low,
                "difference_ci95_high": difference_ci_high,
                "difference_boot_p": difference_boot_p,
                "difference_significant_95ci": (
                    difference_ci_low > 0 or difference_ci_high < 0
                ),
            })

results = pd.DataFrame(rows)
results["conditional_indirect_fdr_p"] = multipletests(
    results["conditional_indirect_boot_p"], method="fdr_bh"
)[1]
results["conditional_indirect_fdr_reject_0.05"] = multipletests(
    results["conditional_indirect_boot_p"], method="fdr_bh"
)[0]

contrasts = pd.DataFrame(contrast_rows)
contrasts["difference_fdr_p"] = multipletests(
    contrasts["difference_boot_p"], method="fdr_bh"
)[1]
contrasts["difference_fdr_reject_0.05"] = multipletests(
    contrasts["difference_boot_p"], method="fdr_bh"
)[0]

RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_TABLE, index=False, na_rep="")
contrasts.to_csv(CONTRAST_TABLE, index=False, na_rep="")

plot_age_moderated_mediation(results, PLOT_DIR / "age_moderated_mediation_forest.svg")

print("Results:", RESULT_TABLE)
print("Indirect-effect age contrasts:", CONTRAST_TABLE)
print("Plots:", PLOT_DIR)
