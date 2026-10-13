#!/usr/bin/env python3
"""Fit two-way moderation and Johnson-Neyman models within school stages."""

import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ========= User settings =========
DATA_TABLE = Path("/path/to/data/final/env_bhv_cov.csv")
SUBJECT_ID, GRADE_COLUMN = "subject_id", "grade"
X_COLUMNS = ["childhood_trauma"]
Y_COLUMNS = ["ms_total_score_grade_z", "cr_total_score_grade_z"]
W_COLUMNS = ["ssrt_reaction_grade_z"]
COVARIATES = ["gender", "f_ses"]
STAGES = {
    "G1-3": [1, 2, 3],
    "G4-6": [4, 5, 6],
    "G7-9": [7, 8, 9],
    "G10-12": [10, 11, 12],
}
MIN_N, ALPHA, DO_FDR = 30, 0.05, True
RESULT_DIR = Path("/path/to/output/stage_moderation_jn")
FIGURE_DIR = Path("/path/to/output/stage_moderation_jn")
# ================================


def safe_name(text):
    for old in ["/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(old, "_")
    return text


def make_model_data(data, x, y, w):
    raw = data[[x, y, w] + COVARIATES].copy()
    covariates = pd.get_dummies(raw[COVARIATES], drop_first=True)
    model_data = pd.concat([raw[[x, y, w]], covariates], axis=1)
    model_data = model_data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    covariate_columns = [column for column in model_data if column not in [x, y, w]]
    model_data["x_c"] = model_data[x] - model_data[x].mean()
    model_data["w_c"] = model_data[w] - model_data[w].mean()
    model_data["xw"] = model_data["x_c"] * model_data["w_c"]
    return model_data, covariate_columns


def fit_model(data, y, covariates):
    predictors = ["x_c", "w_c", "xw"] + covariates
    return sm.OLS(data[y], sm.add_constant(data[predictors], has_constant="add")).fit()


def slope_at(model, w_c):
    beta = model.params["x_c"] + model.params["xw"] * w_c
    cov = model.cov_params()
    se = np.sqrt(cov.loc["x_c", "x_c"] + w_c**2 * cov.loc["xw", "xw"]
                 + 2 * w_c * cov.loc["x_c", "xw"])
    t_value = beta / se
    p_value = 2 * stats.t.sf(abs(t_value), model.df_resid)
    critical = stats.t.ppf(0.975, model.df_resid)
    return beta, se, t_value, p_value, beta - critical * se, beta + critical * se


def plot_model(item, jn_results):
    model_id, stage, x, y, w, model, data, w_mean, w_sd = item
    x_grid = np.linspace(data["x_c"].min(), data["x_c"].max(), 80)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, w_raw, color in [
        ("Low W (-1 SD)", w_mean - w_sd, "#4C72B0"),
        ("Mean W", w_mean, "#55A868"),
        ("High W (+1 SD)", w_mean + w_sd, "#C44E52"),
    ]:
        beta, _, _, p_value, _, _ = slope_at(model, w_raw - w_mean)
        axes[0].plot(x_grid, beta * x_grid, label=f"{label}: b={beta:.3f}, p={p_value:.3g}",
                     color=color, linewidth=2.2)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    axes[0].set(title="Simple slopes", xlabel=f"Centered X: {x}", ylabel="Predicted Y change")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    one_jn = jn_results[jn_results["model_id"] == model_id]
    axes[1].plot(one_jn["moderator_value"], one_jn["conditional_effect"], color="#4C72B0")
    axes[1].fill_between(one_jn["moderator_value"], one_jn["ci95_low"], one_jn["ci95_high"], alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=0.8)
    sig = one_jn["significant_0.05"].to_numpy()
    values = one_jn["moderator_value"].to_numpy()
    start = None
    for i, is_sig in enumerate(sig):
        if is_sig and start is None:
            start = values[i]
        if start is not None and (not is_sig or i == len(sig) - 1):
            end = values[i - 1] if not is_sig else values[i]
            axes[1].axvspan(start, end, color="#55A868", alpha=0.14)
            start = None
    axes[1].set(title="Johnson-Neyman", xlabel=w, ylabel="Conditional effect of X on Y")
    axes[1].text(0.02, 0.98, "band = 95% CI\ngreen = p < .05", transform=axes[1].transAxes,
                 va="top", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"{stage}: {x} x {w} -> {y}", fontsize=11)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{model_id}.png", dpi=300)
    plt.close(fig)


def main():
    data = pd.read_csv(DATA_TABLE)
    needed = [SUBJECT_ID, GRADE_COLUMN] + X_COLUMNS + Y_COLUMNS + W_COLUMNS + COVARIATES
    missing = [column for column in needed if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data[GRADE_COLUMN] = pd.to_numeric(data[GRADE_COLUMN], errors="coerce")
    model_rows, coef_rows, simple_rows, jn_rows, plot_items = [], [], [], [], []

    for stage, grades in STAGES.items():
        stage_data = data[data[GRADE_COLUMN].isin(grades)]
        for x, y, w in product(X_COLUMNS, Y_COLUMNS, W_COLUMNS):
            model_data, covariates = make_model_data(stage_data, x, y, w)
            if len(model_data) < max(MIN_N, len(covariates) + 6):
                print(f"Skipped: {stage}, {y}, n={len(model_data)}")
                continue
            model = fit_model(model_data, y, covariates)
            model_id = safe_name(f"{stage}__{x}__{y}__{w}")
            common = {"model_id": model_id, "stage": stage, "grades": ",".join(map(str, grades)),
                      "x": x, "y": y, "moderator": w, "n": int(model.nobs)}
            model_rows.append({**common, "covariates": "; ".join(covariates), "df_resid": model.df_resid,
                               "r2": model.rsquared, "adj_r2": model.rsquared_adj, "f": model.fvalue,
                               "f_p": model.f_pvalue, "interaction_beta": model.params["xw"],
                               "interaction_se": model.bse["xw"], "interaction_t": model.tvalues["xw"],
                               "interaction_p": model.pvalues["xw"]})
            for term in model.params.index:
                low, high = model.conf_int().loc[term]
                coef_rows.append({**common, "term": term, "beta": model.params[term], "se": model.bse[term],
                                  "t": model.tvalues[term], "p": model.pvalues[term],
                                  "ci95_low": low, "ci95_high": high})

            w_mean, w_sd = model_data[w].mean(), model_data[w].std()
            for label, w_raw in {"low": w_mean - w_sd, "mean": w_mean, "high": w_mean + w_sd}.items():
                beta, se, t_value, p_value, low, high = slope_at(model, w_raw - w_mean)
                simple_rows.append({**common, "moderator_level": label, "moderator_value": w_raw,
                                    "simple_slope": beta, "se": se, "t": t_value, "p": p_value,
                                    "ci95_low": low, "ci95_high": high})
            for w_raw in np.linspace(model_data[w].min(), model_data[w].max(), 200):
                beta, se, _, p_value, low, high = slope_at(model, w_raw - w_mean)
                jn_rows.append({**common, "moderator_value": w_raw, "conditional_effect": beta, "se": se,
                                "p": p_value, "ci95_low": low, "ci95_high": high,
                                "significant_0.05": p_value < ALPHA})
            plot_items.append((model_id, stage, x, y, w, model, model_data, w_mean, w_sd))

    if not model_rows:
        raise ValueError("No stage-specific model met the minimum sample-size requirement")
    models, coefficients = pd.DataFrame(model_rows), pd.DataFrame(coef_rows)
    simple, jn = pd.DataFrame(simple_rows), pd.DataFrame(jn_rows)
    if DO_FDR:
        reject, q_values = multipletests(models["interaction_p"], method="fdr_bh")[:2]
        models["interaction_fdr_p"], models["interaction_fdr_reject_0.05"] = q_values, reject
        q_map = models.set_index("model_id")["interaction_fdr_p"]
        mask = coefficients["term"].eq("xw")
        coefficients.loc[mask, "interaction_fdr_p"] = coefficients.loc[mask, "model_id"].map(q_map)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    models.to_csv(RESULT_DIR / "stage_moderation_model_summary.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(RESULT_DIR / "stage_moderation_coefficients.csv", index=False, encoding="utf-8-sig")
    simple.to_csv(RESULT_DIR / "stage_moderation_simple_slopes.csv", index=False, encoding="utf-8-sig")
    jn.to_csv(RESULT_DIR / "stage_moderation_johnson_neyman.csv", index=False, encoding="utf-8-sig")
    for item in plot_items:
        plot_model(item, jn)
    print(f"Done: stage-specific moderation results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
