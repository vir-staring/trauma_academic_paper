#!/usr/bin/env python3
"""Fit the three-way moderation model separately within four school stages."""

import importlib.util
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/path/to/output/cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


# ========= User settings =========
FILE_MAIN = "/path/to/data/final/env_bhv_cov.csv"  # 环境/行为/保护因素/协变量
FILE_BRAIN = "/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv"
ID_COL, GRADE_COL = "subject_id", "grade"

ENV_COLS = ["childhood_trauma"]
BEHAV_COLS = [
    "ms_total_score_grade_z", 
    "cr_total_score_grade_z"
    ]
BRAIN_COLS = [    
    "Cingulo-Opercular_FA_grade_z",
    # "Frontoparietal_FA_grade_z",
    # "Posterior-Multimodal_FA_grade_z",
    # "Language_FA_grade_z",
    # "Dorsal-Attention_FA_grade_z"
    ]
PROT_COLS = ["ssrt_reaction_grade_z"]
COVARIATES = ["gender", "f_ses"]

STAGES = {
    "G1-3": [1, 2, 3],
    "G4-6": [4, 5, 6],
    "G7-9": [7, 8, 9],
    "G10-12": [10, 11, 12],
}
MIN_N, ALPHA = 30, 0.05
USE_FDR, PLOT_SIGNIFICANT = True, True

OUT_DIR = Path("/path/to/output/stage_double_moderation")
PLOT_DIR = Path("/path/to/output/stage_double_moderation")
# ================================


base_path = Path(__file__).with_name("8_double_moderated_env-bhv.py")
spec = importlib.util.spec_from_file_location("double_moderation_base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.COVARIATES, base.MIN_N, base.ALPHA = COVARIATES, MIN_N, ALPHA


def plot_jn_fix_w(model, data, x, y, b, w, stage):
    """Fix W at +/-1 SD and plot the conditional X effect across B."""
    exog = model.model.exog_names
    b_values = np.linspace(data[b].min(), data[b].max(), 100)
    critical_t = t_dist.ppf(0.975, model.df_resid)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for w_level, label in [(-1.0, "W -1 SD"), (1.0, "W +1 SD")]:
        effects, ses = [], []
        for b_level in b_values:
            weights = np.zeros(len(exog))
            weights[exog.index(x)] = 1
            weights[exog.index("XB")] = b_level
            weights[exog.index("XW")] = w_level
            weights[exog.index("XBW")] = b_level * w_level
            estimate, se, *_ = base.contrast_stats(model, weights)
            effects.append(estimate)
            ses.append(se)
        effects, ses = np.asarray(effects), np.asarray(ses)
        ax.plot(b_values, effects, label=label)
        ax.fill_between(b_values, effects - critical_t * ses, effects + critical_t * ses, alpha=0.15)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(title=f"{stage}: {x} -> {y}\nW fixed at +/-1 SD", xlabel=b,
           ylabel=f"Conditional effect of {x}")
    ax.legend(frameon=False)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    name = "__".join([stage, x, y, b, w]).replace("/", "_")
    fig.savefig(PLOT_DIR / f"JN_fixW__{name}.png", dpi=300)
    plt.close(fig)


def main():
    main_df = pd.read_csv(FILE_MAIN)
    brain_df = pd.read_csv(FILE_BRAIN)
    main_cols = [ID_COL, GRADE_COL] + ENV_COLS + BEHAV_COLS + PROT_COLS + COVARIATES
    brain_cols = [ID_COL] + BRAIN_COLS
    missing_main = [column for column in main_cols if column not in main_df]
    missing_brain = [column for column in brain_cols if column not in brain_df]
    if missing_main or missing_brain:
        raise ValueError(f"Missing columns: main={missing_main}, brain={missing_brain}")

    df = main_df[main_cols].merge(brain_df[brain_cols], on=ID_COL, how="inner")
    df[GRADE_COL] = pd.to_numeric(df[GRADE_COL], errors="coerce")
    combinations = list(product(ENV_COLS, BEHAV_COLS, BRAIN_COLS, PROT_COLS))
    summaries, coefficient_rows, conditional_rows, fitted = [], [], [], {}
    total = len(STAGES) * len(combinations)
    count = 0

    for stage, grades in STAGES.items():
        stage_df = df[df[GRADE_COL].isin(grades)].copy()
        for x, y, b, w in combinations:
            count += 1
            print(f"[{count}/{total}] stage={stage}, X={x}, Y={y}, B={b}, W={w}")
            try:
                model, data, summary, coefs, conditions = base.run_model(stage_df, x, y, b, w)
            except ValueError as error:
                print(f"Skipped: {error}")
                continue

            model_id = f"{stage}__{summary['model_id']}"
            summary.update({"model_id": model_id, "stage": stage, "grades": ",".join(map(str, grades))})
            for row in coefs + conditions:
                row.update({"model_id": model_id, "stage": stage, "grades": summary["grades"]})
            summaries.append(summary)
            coefficient_rows.extend(coefs)
            conditional_rows.extend(conditions)
            fitted[model_id] = (model, data, x, y, b, w, stage)

    if not summaries:
        raise ValueError("No stage-specific model met the minimum sample-size requirement")

    full = pd.DataFrame(summaries)
    coefficients = pd.DataFrame(coefficient_rows)
    conditional = pd.DataFrame(conditional_rows)
    base.add_fdr(full, coefficients)
    for suffix in ["XB", "XW", "XBW"]:
        test = full[f"q_{suffix}"] if USE_FDR else full[f"p_{suffix}"]
        full[f"sig_{suffix}"] = np.where(
            test < ALPHA, np.where(full[f"beta_{suffix}"] > 0, "Yes (+)", "Yes (-)"), "No"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_DIR / "stage_double_moderation_full_results.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(OUT_DIR / "stage_double_moderation_coefficient_results.csv", index=False,
                        encoding="utf-8-sig")
    conditional.to_csv(OUT_DIR / "stage_double_moderation_conditional_effects.csv", index=False,
                       encoding="utf-8-sig")

    paper = base.make_paper_table(full, coefficients, conditional)
    paper = paper.merge(full[["model_id", "stage", "grades"]], on="model_id", how="left")
    paper = paper[["model_id", "stage", "grades"] + [c for c in paper if c not in ["model_id", "stage", "grades"]]]
    paper.to_csv(OUT_DIR / "stage_double_moderation_paper_key_results.csv", index=False, encoding="utf-8-sig")

    sig_cols = ["model_id", "stage", "grades", "env", "behav", "brain", "prot", "n", "R2", "adjusted_R2"]
    for suffix in ["XB", "XW", "XBW"]:
        sig_cols += [f"beta_{suffix}", f"se_{suffix}", f"ci95_lower_{suffix}", f"ci95_upper_{suffix}",
                     f"p_{suffix}", f"q_{suffix}", f"sig_{suffix}"]
    full[sig_cols].to_csv(OUT_DIR / "stage_double_moderation_significant_summary.csv", index=False,
                          encoding="utf-8-sig")

    if PLOT_SIGNIFICANT:
        test_col = "q_XBW" if USE_FDR else "p_XBW"
        for model_id in full.loc[full[test_col] < ALPHA, "model_id"]:
            plot_jn_fix_w(*fitted[model_id])
    print(f"Done: stage-specific results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
