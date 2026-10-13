"""
调节效应分析：环境→行为，白质束脑指标作为调节变量，控制年龄性别
- 线性回归: behav ~ env + brain + env*brain + age + sex
- 仅报告显著调节效应(交互项)，绘制简单斜率图 + Johnson-Neyman图
- 支持FDR多重比较校正开关
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
from itertools import product
from pathlib import Path

plt.rcParams['font.family'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===================== 用户配置 =====================
BEHAV_FILE =  "/path/to/data/final/trauma_academic_ssrt_covs.csv"             # 环境/行为/协变量表（第一列为被试编号）
BRAIN_FILE = "/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv"
OUTPUT_DIR = Path("/path/to/output/moderate")    # 输出目录

# 列名配置
ENV_COLS   = [
    "childhood_trauma"
    ]   # 自变量(环境)
BEHAV_COLS = [
    "ms_total_score_grade_z",
    "cr_total_score_grade_z"
    ]   # 因变量(行为)
BRAIN_COLS = [
   "ssrt_reaction_grade_z",
    ]    # 调节变量(脑指标) — 用户指定关注的列

COV_COLS   = [
    "gender", 
    "f_ses", 
    ]             # 协变量(控制变量)

# 分析参数
ALPHA       = 0.05
USE_FDR     = False   # 是否进行FDR多重比较校正
STANDARDIZE = True   # 是否对连续变量标准化
# ====================================================


def moderation_test(df, env, behav, mod, covs):
    """behav ~ env + mod + env*mod + covs, 返回交互项统计量"""
    cols = [behav, env, mod] + covs
    d = df[cols].dropna()
    if len(d) < len(cols) + 5:
        return None
    y = d[behav].values
    interaction = d[env].values * d[mod].values
    X = sm.add_constant(np.column_stack([d[env], d[mod], interaction, d[covs].values]))
    try:
        m = sm.OLS(y, X).fit()
    except Exception:
        return None
    return {"b": m.params[3], "se": m.bse[3], "t": m.tvalues[3], "p": m.pvalues[3],
            "R2": m.rsquared, "n": len(d), "model": m, "data": d}


def johnson_neyman(model, data, env, mod, alpha=0.05):
    """返回调节变量值域上的条件效应、CI及JN转折点"""
    b1, b3 = model.params[1], model.params[3]
    C = model.cov_params()
    t_c = stats.t.ppf(1 - alpha / 2, model.df_resid)
    mv = np.linspace(data[mod].min(), data[mod].max(), 500)
    slopes = b1 + b3 * mv
    se = np.sqrt(C[1, 1] + 2 * mv * C[1, 3] + mv**2 * C[3, 3])
    sig = np.abs(slopes / se) > t_c
    jn = [mv[i] for i in range(1, len(sig)) if sig[i] != sig[i - 1]]
    return mv, slopes, se, t_c, jn


def plot_mod(data, env, behav, mod, mv, slopes, se, t_c, jn, out_path):
    """简单斜率图(左) + Johnson-Neyman图(右)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 简单斜率 (M±1SD)
    mu, sd = data[mod].mean(), data[mod].std()
    xr = np.linspace(data[env].min(), data[env].max(), 100)
    for lv, lb, c in [(-1, "Low(-1SD)", "#4575b4"), (0, "Mean", "gray"), (1, "High(+1SD)", "#d73027")]:
        s = slopes[np.argmin(np.abs(mv - (mu + lv * sd)))]
        ax1.plot(xr, s * xr, label=f"{lb}", color=c, lw=2)
    ax1.axhline(0, color="k", lw=.5, ls="--")
    ax1.set(xlabel=env, ylabel=behav, title=f"Simple Slopes: {env}→{behav} | {mod}")
    ax1.legend(fontsize=9, loc="upper left")

       # 右: JN图
    ci_u, ci_l = slopes + t_c * se, slopes - t_c * se
    sig_mask = (ci_u < 0) | (ci_l > 0)  # CI不包含0 → 显著
    ax2.plot(mv, slopes, "k-", lw=2, label="Conditional effect")
    ax2.fill_between(mv, ci_l, ci_u, alpha=.2, color="gray", label="95% CI")
    ax2.axhline(0, color="red", lw=1, ls="--")
    # 显著区域底色（用 axvspan 标记连续显著段）
    in_sig, labeled = False, False
    for i in range(len(sig_mask)):
        if sig_mask[i] and not in_sig:
            start = mv[i]; in_sig = True
        elif not sig_mask[i] and in_sig:
            ax2.axvspan(start, mv[i - 1], alpha=0.12, color="#2ca02c",
                        label="Significant region" if not labeled else "")
            in_sig = False; labeled = True
    if in_sig:
        ax2.axvspan(start, mv[-1], alpha=0.12, color="#2ca02c",
                    label="Significant region" if not labeled else "")
    for j in jn:
        ax2.axvline(j, color="blue", lw=1.5, ls=":", label=f"JN={j:.3f}")
    ax2.set(xlabel=f"Moderator: {mod}", ylabel="Conditional effect", title="Johnson-Neyman")
    ax1.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================== 主程序 =====================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 读取 & 合并（自动统一被试编号列名）
    SUBJ = "subject_id"  # 统一被试编号列名
    behav_df = pd.read_csv(BEHAV_FILE)
    behav_df = behav_df.rename(columns={behav_df.columns[0]: SUBJ})
    brain_df = pd.read_csv(BRAIN_FILE)
    brain_df = brain_df.rename(columns={brain_df.columns[0]: SUBJ})
    df = pd.merge(behav_df, brain_df, on=SUBJ, how="inner")
    print(f"合并后被试数: {len(df)}")

    # 标准化连续变量
    if STANDARDIZE:
        num_cols = [c for c in ENV_COLS + BEHAV_COLS + BRAIN_COLS + COV_COLS
                    if df[c].dtype in ("float64", "float32", "int64") and df[c].nunique() > 5]
        df[num_cols] = df[num_cols].apply(lambda x: stats.zscore(x, nan_policy="omit"))

    # 遍历所有 env × behav × brain 组合
    records = []
    for env, beh, br in product(ENV_COLS, BEHAV_COLS, BRAIN_COLS):
        r = moderation_test(df, env, beh, br, COV_COLS)
        if r:
            records.append({"env": env, "behav": beh, "brain": br, **r})

    if not records:
        print("无有效结果"); exit()

    res = pd.DataFrame(records)

    # FDR校正
    if USE_FDR:
        _, res["p_fdr"], _, _ = multipletests(res["p"], method="fdr_bh")
        sig_col, sig_thr = "p_fdr", ALPHA
        print(f"已进行FDR校正，共 {len(res)} 个检验")
    else:
        sig_col, sig_thr = "p", ALPHA

    # 筛选显著结果
    sig = res[res[sig_col] < sig_thr].copy()
    print(f"显著调节效应: {len(sig)} / {len(res)}")

    # 保存全部结果
    out_cols = ["env", "behav", "brain", "b", "se", "t", "p", "R2", "n"]
    if USE_FDR:
        out_cols.append("p_fdr")
    res[out_cols].sort_values(sig_col).to_csv(OUTPUT_DIR / "all_results.csv", index=False, encoding="utf-8-sig")

    # 保存显著结果 & 绘图
    if len(sig) > 0:
        sig[out_cols].to_csv(OUTPUT_DIR / "significant_results.csv", index=False, encoding="utf-8-sig")
        for _, row in sig.iterrows():
            mv, slopes, se, t_c, jn = johnson_neyman(row["model"], row["data"], row["env"], row["brain"])
            fig_name = f"{row['env']}__{row['behav']}__{row['brain']}.png".replace("/", "_")
            plot_mod(row["data"], row["env"], row["behav"], row["brain"],
                     mv, slopes, se, t_c, jn, OUTPUT_DIR / fig_name)
            print(f"  绘图: {fig_name} | JN points: {[f'{j:.3f}' for j in jn]}")
    else:
        print("无显著调节效应，不绘图")

    print("完成!")