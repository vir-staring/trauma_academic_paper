import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("MPLCONFIGDIR", "/path/to/output/mpl_cache")

import matplotlib.pyplot as plt
from itertools import product
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests

plt.rcParams["svg.fonttype"] = "none"


# ========= 用户自定义参数 =========
FILE_MAIN = "/path/to/data/final/env_bhv_cov.csv"  # 环境/行为/保护因素/协变量
FILE_BRAIN = "/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv"
# FILE_BRAIN = "/path/to/data/dmri_meas_harmonized_site_qc_z.csv"
PLOT_DIR = Path("/path/to/output/double_moderated/double_moderation_plots")
OUT_FULL = "/path/to/output/double_moderated/double_moderation_full_results.csv"
OUT_SIG = "/path/to/output/double_moderated/double_moderation_significant_summary.csv"
OUT_COEFS = "/path/to/output/double_moderated/double_moderation_coefficient_results.csv"
OUT_CONDITIONAL = "/path/to/output/double_moderated/double_moderation_conditional_effects.csv"
OUT_PAPER = "/path/to/output/double_moderated/double_moderation_paper_key_results.csv"

ID_COL = "subject_id"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True

ENV_COLS   = [
    "childhood_trauma"
    ]   # 自变量(环境)
BEHAV_COLS = [
    "ms_total_score_grade_z",
    "cr_total_score_grade_z"
    ]   # 因变量(行为)
BRAIN_COLS = [
    "Cingulo-Opercular_FA_grade_z",
    "Frontoparietal_FA_grade_z",
    "Posterior-Multimodal_FA_grade_z",
    "Language_FA_grade_z",
    "Dorsal-Attention_FA_grade_z"
    ]    # 调节变量(脑指标) — 用户指定关注的列
PROT_COLS = [
    "ssrt_reaction_grade_z",
]                    # W: 保护因素(调节X->M)
COVARIATES   = [
    "gender", 
    "f_ses", 
]             # 协变量(控制变量)


MIN_N = 30

PLOT_COLORS = {
    "brain_low": "#3F75B5",
    "brain_high": "#D35D60",
}
OUTCOME_LABELS = {
    "ms_total_score_grade_z": "Mathematics",
    "cr_total_score_grade_z": "Chinese reading",
}
NETWORK_LABELS = {
    "Cingulo-Opercular_FA_grade_z": "Cingulo-opercular",
    "Frontoparietal_FA_grade_z": "Frontoparietal",
    "Posterior-Multimodal_FA_grade_z": "Posterior multimodal",
    "Language_FA_grade_z": "Language",
    "Dorsal-Attention_FA_grade_z": "Dorsal attention",
}


# 显著性设置
USE_FDR = True           # True: 使用FDR矫正后的q值判显著；False: 使用原始p值
ALPHA = 0.05

# 绘图设置
PLOT_SIGNIFICANT = True

PLOT_WHEN = "XBW"       # 可选: "XBW"(仅三重交互显著) 或 "ANY"(XB/XW/XBW任一显著)
# ===============================


def zscore_continuous(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().nunique() <= 2:
        return series
    sd = series.std(ddof=0)
    return (series - series.mean()) / sd if sd > 0 else series


def fit_model(df, x, y, b, w):
    raw = df[[x, y, b, w] + COVARIATES].copy()
    covariates = pd.get_dummies(raw[COVARIATES], drop_first=True)
    data = pd.concat([raw[[x, y, b, w]], covariates], axis=1).dropna().reset_index(drop=True)
    cov_terms = [c for c in data.columns if c not in [x, y, b, w]]
    if len(data) < max(MIN_N, len(cov_terms) + 10):
        return None, None

    for column in data.columns:
        data[column] = zscore_continuous(data[column])
    data["XB"] = data[x] * data[b]
    data["XW"] = data[x] * data[w]
    data["BW"] = data[b] * data[w]
    data["XBW"] = data[x] * data[b] * data[w]
    predictors = [x, b, w, "XB", "XW", "BW", "XBW"] + cov_terms
    model = sm.OLS(data[y], sm.add_constant(data[predictors], has_constant="add")).fit()
    return model, data


def contrast_stats(model, weights):
    weights = np.asarray(weights, dtype=float)
    estimate = float(weights @ model.params.to_numpy())
    se = float(np.sqrt(weights @ model.cov_params().to_numpy() @ weights))
    t_value = estimate / se if se > 0 else np.nan
    p_value = 2 * t_dist.sf(abs(t_value), model.df_resid) if np.isfinite(t_value) else np.nan
    critical_t = t_dist.ppf(0.975, model.df_resid)
    return estimate, se, t_value, estimate - critical_t * se, estimate + critical_t * se, p_value


def run_model(df, x, y, b, w):
    model, data = fit_model(df, x, y, b, w)
    if model is None:
        raise ValueError(f"样本量不足: X={x}, Y={y}, B={b}, W={w}")

    model_id = "__".join([x, y, b, w])
    exog = model.model.exog_names
    ci = model.conf_int()
    roles = {
        "const": "Intercept", x: "X", b: "B", w: "W",
        "XB": "X × B", "XW": "X × W", "BW": "B × W", "XBW": "X × B × W",
    }
    summary = {
        "model_id": model_id,
        "model_formula": y + " ~ " + " + ".join(exog[1:]),
        "env": x, "behav": y, "brain": b, "prot": w,
        "covariates": ";".join(COVARIATES),
        "scaling": "continuous variables z-scored (ddof=0); binary variables unchanged",
        "covariance_type": model.cov_type,
        "n_merged": len(df), "n": len(data), "n_excluded": len(df) - len(data),
        "df_model": model.df_model, "df_resid": model.df_resid,
        "R2": model.rsquared, "adjusted_R2": model.rsquared_adj,
        "F_statistic": model.fvalue, "F_p_value": model.f_pvalue,
        "AIC": model.aic, "BIC": model.bic,
        "RMSE": float(np.sqrt(np.mean(model.resid**2))),
    }

    coefficient_rows = []
    suffixes = {x: "X", b: "B", w: "W", "XB": "XB", "XW": "XW", "BW": "BW", "XBW": "XBW"}
    for term in exog:
        row = {
            "model_id": model_id, "env": x, "behav": y, "brain": b, "prot": w, "n": len(data),
            "term": term, "term_label": roles.get(term, term),
            "estimate": model.params[term], "std_error": model.bse[term], "t_value": model.tvalues[term],
            "ci95_lower": ci.loc[term, 0], "ci95_upper": ci.loc[term, 1], "p_value": model.pvalues[term],
        }
        coefficient_rows.append(row)
        if term in suffixes:
            suffix = suffixes[term]
            for key, value in [
                ("beta", row["estimate"]), ("se", row["std_error"]), ("t", row["t_value"]),
                ("ci95_lower", row["ci95_lower"]), ("ci95_upper", row["ci95_upper"]), ("p", row["p_value"]),
            ]:
                summary[f"{key}_{suffix}"] = value

    conditional_rows = []
    for b_level, w_level in product([-1.0, 1.0], repeat=2):
        weights = np.zeros(len(exog))
        weights[exog.index(x)] = 1
        weights[exog.index("XB")] = b_level
        weights[exog.index("XW")] = w_level
        weights[exog.index("XBW")] = b_level * w_level
        estimate, se, t_value, ci_low, ci_high, p_value = contrast_stats(model, weights)
        label = ("Bhigh" if b_level > 0 else "Blow") + "_" + ("Whigh" if w_level > 0 else "Wlow")
        summary.update({
            f"x_eff_{label}": estimate, f"se_x_eff_{label}": se,
            f"ci95_lower_x_eff_{label}": ci_low, f"ci95_upper_x_eff_{label}": ci_high,
            f"p_x_eff_{label}": p_value,
        })
        conditional_rows.append({
            "model_id": model_id, "env": x, "behav": y, "brain": b, "prot": w, "n": len(data),
            "brain_level": "high (+1 SD)" if b_level > 0 else "low (-1 SD)",
            "prot_level": "high (+1 SD)" if w_level > 0 else "low (-1 SD)",
            "conditional_X_effect": estimate, "std_error": se, "t_value": t_value,
            "ci95_lower": ci_low, "ci95_upper": ci_high, "p_value": p_value,
        })
    return model, data, summary, coefficient_rows, conditional_rows


def add_fdr(full, coefficients):
    coefficients["fdr_q"] = np.nan
    coefficients["fdr_reject_0.05"] = False
    for suffix, term in [("XB", "XB"), ("XW", "XW"), ("XBW", "XBW")]:
        full[f"q_{suffix}"] = multipletests(full[f"p_{suffix}"], method="fdr_bh")[1]
        q_lookup = full.set_index("model_id")[f"q_{suffix}"]
        mask = coefficients["term"].eq(term)
        coefficients.loc[mask, "fdr_q"] = coefficients.loc[mask, "model_id"].map(q_lookup)
        coefficients.loc[mask, "fdr_reject_0.05"] = coefficients.loc[mask, "fdr_q"] < ALPHA


def make_paper_table(full, coefficients, conditional):
    key_terms = ["X", "B", "W", "X × B", "X × W", "B × W", "X × B × W"]
    coef = coefficients[coefficients["term_label"].isin(key_terms)].copy()
    coef["section"] = "Model coefficient"
    coef["condition"] = ""
    coef = coef.rename(columns={"estimate": "coefficient_B"})

    cond = conditional.copy()
    cond["section"] = "Conditional X effect"
    cond["term_label"] = "Conditional X effect"
    cond["condition"] = "B " + cond["brain_level"] + ", W " + cond["prot_level"]
    cond["fdr_q"] = np.nan
    cond["fdr_reject_0.05"] = False
    cond = cond.rename(columns={"conditional_X_effect": "coefficient_B"})

    result_cols = [
        "model_id", "env", "behav", "brain", "prot", "n", "section", "term_label", "condition",
        "coefficient_B", "std_error", "t_value", "ci95_lower", "ci95_upper", "p_value",
        "fdr_q", "fdr_reject_0.05",
    ]
    paper = pd.concat([coef[result_cols], cond[result_cols]], ignore_index=True)
    model_cols = [
        "model_id", "model_formula", "covariates", "scaling", "R2", "adjusted_R2",
        "df_model", "df_resid", "F_statistic", "F_p_value", "AIC", "BIC",
    ]
    paper = paper.merge(full[model_cols], on="model_id", how="left")
    paper["F_df"] = [f"F({int(a)}, {int(b)})" for a, b in zip(paper["df_model"], paper["df_resid"])]
    paper["decision_metric"] = np.where(paper["fdr_q"].notna(), "BH-FDR q", "raw p")
    paper["significant_0.05"] = np.where(
        paper["fdr_q"].notna(), paper["fdr_q"] < ALPHA, paper["p_value"] < ALPHA
    )
    columns = [
        "model_id", "behav", "env", "brain", "prot", "covariates", "model_formula", "scaling", "n",
        "R2", "adjusted_R2", "F_df", "F_statistic", "F_p_value", "AIC", "BIC", "section",
        "term_label", "condition", "coefficient_B", "std_error", "t_value", "ci95_lower", "ci95_upper",
        "p_value", "fdr_q", "decision_metric", "significant_0.05",
    ]
    return paper[columns]


def plot_conditional_x_effect_fix_brain(model, data, x, y, b, w, plot_dir=PLOT_DIR):
    """Fix brain FA at ±1 SD and plot the conditional X effect across SSRT."""
    exog = model.model.exog_names
    w_values = np.linspace(data[w].min(), data[w].max(), 100)
    critical_t = t_dist.ppf(0.975, model.df_resid)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for b_level, label, color in [
        (-1.0, "Brain FA -1 SD", PLOT_COLORS["brain_low"]),
        (1.0, "Brain FA +1 SD", PLOT_COLORS["brain_high"]),
    ]:
        effects, ses = [], []
        for w_level in w_values:
            weights = np.zeros(len(exog))
            weights[exog.index(x)] = 1
            weights[exog.index("XB")] = b_level
            weights[exog.index("XW")] = w_level
            weights[exog.index("XBW")] = b_level * w_level
            estimate, se, *_ = contrast_stats(model, weights)
            effects.append(estimate)
            ses.append(se)
        effects, ses = np.asarray(effects), np.asarray(ses)
        ax.plot(w_values, effects, color=color, linewidth=2.8)
        ax.fill_between(
            w_values,
            effects - critical_t * ses,
            effects + critical_t * ses,
            color=color,
            alpha=.18,
        )

    handles = [Line2D([], [], marker="o", linestyle="", markersize=11,
                      markerfacecolor=PLOT_COLORS[key], markeredgecolor="none", label=label)
               for key, label in [("brain_low", "Brain FA −1 SD"),
                                  ("brain_high", "Brain FA +1 SD")]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.19),
              ncol=2, frameon=False, fontsize=18, columnspacing=3)
    ax.text(.5, 1.38, f"{NETWORK_LABELS[b]} network FA",
            transform=ax.transAxes, ha="center", fontsize=24, fontweight="bold")
    ax.text(-.02, 1.08, "Conditional trauma–academic association (b) with 95% CI",
            transform=ax.transAxes, fontsize=19, color="#666B73")
    ax.set_xlabel("SSRT", fontsize=21, labelpad=20)
    ax.tick_params(labelsize=17)
    ax.axhline(0, color="#E4E7EB", linewidth=.8)
    ax.grid(axis="y", color="#E4E7EB", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A8A8A")
    fig.suptitle(OUTCOME_LABELS[y], x=.54, y=.975, fontsize=31, fontweight="bold")
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.65)
    plot_dir.mkdir(parents=True, exist_ok=True)
    name = "__".join([x, y, b, w]).replace("/", "_")
    fig.savefig(plot_dir / f"conditional_X_effect_fixBrain__{name}.svg",
                dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def make_demo_data(n=3600):
    rng = np.random.default_rng(20260909)
    x, w = rng.normal(size=(2, n))
    brains = {column: rng.normal(size=n) for column in BRAIN_COLS}
    gender = rng.binomial(1, .47, n)
    ses = rng.normal(size=n)
    base = -.35 * x - .18 * w + .08 * gender + .16 * ses
    reading_xbw = sum(-.10 * x * w * brain for brain in brains.values())
    return pd.DataFrame({
        ID_COL: [f"demo_{index:04d}" for index in range(n)],
        ENV_COLS[0]: x,
        PROT_COLS[0]: w,
        COVARIATES[0]: gender,
        COVARIATES[1]: ses,
        BEHAV_COLS[0]: base - .12 * x * w * brains["Dorsal-Attention_FA_grade_z"] + rng.normal(size=n),
        BEHAV_COLS[1]: base + reading_xbw + rng.normal(size=n),
        **brains,
    })


def main():
    demo = "--demo" in sys.argv[1:]
    if demo:
        source = make_demo_data()
        main_df, brain_df = source, source
        output_root = Path("/path/to/output/double_moderation_demo")
        plot_dir = output_root / "plots"
        out_full = output_root / "double_moderation_full_results.csv"
        out_sig = output_root / "double_moderation_significant_summary.csv"
        out_coefs = output_root / "double_moderation_coefficient_results.csv"
        out_conditional = output_root / "double_moderation_conditional_effects.csv"
        out_paper = output_root / "double_moderation_paper_key_results.csv"
    else:
        main_df, brain_df = pd.read_csv(FILE_MAIN), pd.read_csv(FILE_BRAIN)
        plot_dir = PLOT_DIR
        out_full, out_sig = Path(OUT_FULL), Path(OUT_SIG)
        out_coefs, out_conditional, out_paper = map(Path, [OUT_COEFS, OUT_CONDITIONAL, OUT_PAPER])

    main_cols = [ID_COL] + ENV_COLS + BEHAV_COLS + PROT_COLS + COVARIATES
    brain_cols = [ID_COL] + BRAIN_COLS
    missing_main = [c for c in main_cols if c not in main_df]
    missing_brain = [c for c in brain_cols if c not in brain_df]
    if missing_main or missing_brain:
        raise ValueError(f"列缺失: main={missing_main}, brain={missing_brain}")
    df = main_df[main_cols].merge(brain_df[brain_cols], on=ID_COL, how="inner")

    summaries, coefficient_rows, conditional_rows, fitted = [], [], [], {}
    combinations = list(product(ENV_COLS, BEHAV_COLS, BRAIN_COLS, PROT_COLS))
    for i, (x, y, b, w) in enumerate(combinations, 1):
        print(f"[{i}/{len(combinations)}] X={x}, Y={y}, B={b}, W={w}")
        model, data, summary, coefs, conditions = run_model(df, x, y, b, w)
        summaries.append(summary)
        coefficient_rows.extend(coefs)
        conditional_rows.extend(conditions)
        fitted[summary["model_id"]] = (model, data, x, y, b, w)

    full = pd.DataFrame(summaries)
    coefficients = pd.DataFrame(coefficient_rows)
    conditional = pd.DataFrame(conditional_rows)
    add_fdr(full, coefficients)

    for suffix in ["XB", "XW", "XBW"]:
        test = full[f"q_{suffix}"] if USE_FDR else full[f"p_{suffix}"]
        full[f"sig_{suffix}"] = np.where(test < ALPHA, np.where(full[f"beta_{suffix}"] > 0, "Yes (+)", "Yes (-)"), "No")

    out_full.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_full, index=False, encoding="utf-8-sig", float_format="%.6g")
    coefficients.to_csv(out_coefs, index=False, encoding="utf-8-sig", float_format="%.6g")
    conditional.to_csv(out_conditional, index=False, encoding="utf-8-sig", float_format="%.6g")
    make_paper_table(full, coefficients, conditional).to_csv(
        out_paper, index=False, encoding="utf-8-sig", float_format="%.6g"
    )

    sig_cols = ["model_id", "env", "behav", "brain", "prot", "n", "R2", "adjusted_R2"]
    for suffix in ["XB", "XW", "XBW"]:
        sig_cols += [
            f"beta_{suffix}", f"se_{suffix}", f"ci95_lower_{suffix}", f"ci95_upper_{suffix}",
            f"p_{suffix}", f"q_{suffix}", f"sig_{suffix}",
        ]
    full[sig_cols].to_csv(out_sig, index=False, encoding="utf-8-sig", float_format="%.6g")

    if PLOT_SIGNIFICANT:
        test_col = "q_XBW" if USE_FDR else "p_XBW"
        for _, row in full[full[test_col] < ALPHA].iterrows():
            plot_conditional_x_effect_fix_brain(*fitted[row["model_id"]], plot_dir=plot_dir)

    print(f"完成: {out_full}, {out_sig}, {out_coefs}, {out_conditional}, {out_paper}")


if __name__ == "__main__":
    main()
