#!/usr/bin/env python3
"""Add SES PC1 score to the environment table."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TABLE = Path("/path/to/data/processed/envs.csv")
SES_COLUMNS = ["Parent occupation", "Parent education"]
SES_SCORE = "ses_pc1"
FIGURE = Path("/path/to/data/processed/figures/ses_pca.png")
REPORT = Path("/path/to/data/processed/log/ses_pca_report.txt")


def get_separator(path):
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


df = pd.read_csv(TABLE, sep=get_separator(TABLE))
ses = df[SES_COLUMNS].apply(pd.to_numeric, errors="coerce")
complete = ses.dropna()

# PCA 只使用两个 SES 原始变量都不缺失的被试。
scaler = StandardScaler()
scaled = scaler.fit_transform(complete)
pca = PCA(n_components=2)
scores = pca.fit_transform(scaled)
pc1 = scores[:, 0]

# 让 SES 分数方向更直观：受教育程度越高，ses_pc1 越高。
if pd.Series(pc1).corr(complete[SES_COLUMNS[0]]) < 0:
    pc1 = -pc1
    pca.components_[0] = -pca.components_[0]

# 只给 PCA 有效样本写入 ses_pc1，原始缺失样本保持为空。
df[SES_SCORE] = pd.NA
df.loc[complete.index, SES_SCORE] = pc1
df.to_csv(TABLE, sep=get_separator(TABLE), index=False, na_rep="")

FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].bar(["PC1", "PC2"], pca.explained_variance_ratio_)
axes[0].set_title("Explained variance")
axes[0].set_ylabel("Ratio")
axes[1].bar(["education", "occupation"], pca.components_[0])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("PC1 loadings")
axes[2].hist(pc1, bins=25, edgecolor="black")
axes[2].set_title("SES PC1 distribution")
axes[2].set_xlabel(SES_SCORE)
axes[2].set_ylabel("Count")
fig.tight_layout()
fig.savefig(FIGURE, dpi=300)
plt.close(fig)

REPORT.parent.mkdir(parents=True, exist_ok=True)
pc1_series = pd.Series(pc1, index=complete.index)
lines = [
    "SES PCA report",
    "",
    "Input table: " + str(TABLE),
    "SES columns: " + ", ".join(SES_COLUMNS),
    "SES score column: " + SES_SCORE,
    "",
    "Sample size",
    "total_rows: " + str(len(df)),
    "complete_rows_used_for_pca: " + str(len(complete)),
    "missing_" + SES_COLUMNS[0] + ": " + str(ses[SES_COLUMNS[0]].isna().sum()),
    "missing_" + SES_COLUMNS[1] + ": " + str(ses[SES_COLUMNS[1]].isna().sum()),
    "",
    "Scaler",
    SES_COLUMNS[0] + "_mean: " + str(round(scaler.mean_[0], 6)),
    SES_COLUMNS[0] + "_scale: " + str(round(scaler.scale_[0], 6)),
    SES_COLUMNS[1] + "_mean: " + str(round(scaler.mean_[1], 6)),
    SES_COLUMNS[1] + "_scale: " + str(round(scaler.scale_[1], 6)),
    "",
    "PCA model",
    "n_components: " + str(pca.n_components_),
    "n_features_in: " + str(pca.n_features_in_),
    "n_samples: " + str(pca.n_samples_),
    "explained_variance_PC1: " + str(round(pca.explained_variance_[0], 6)),
    "explained_variance_PC2: " + str(round(pca.explained_variance_[1], 6)),
    "explained_variance_ratio_PC1: " + str(round(pca.explained_variance_ratio_[0], 6)),
    "explained_variance_ratio_PC2: " + str(round(pca.explained_variance_ratio_[1], 6)),
    "singular_value_PC1: " + str(round(pca.singular_values_[0], 6)),
    "singular_value_PC2: " + str(round(pca.singular_values_[1], 6)),
    "noise_variance: " + str(round(pca.noise_variance_, 6)),
    "",
    "PC1 loadings",
    SES_COLUMNS[0] + ": " + str(round(pca.components_[0, 0], 6)),
    SES_COLUMNS[1] + ": " + str(round(pca.components_[0, 1], 6)),
    "",
    "PC2 loadings",
    SES_COLUMNS[0] + ": " + str(round(pca.components_[1, 0], 6)),
    SES_COLUMNS[1] + ": " + str(round(pca.components_[1, 1], 6)),
    "",
    "Correlation with SES PC1",
    SES_COLUMNS[0] + ": " + str(round(pc1_series.corr(complete[SES_COLUMNS[0]]), 6)),
    SES_COLUMNS[1] + ": " + str(round(pc1_series.corr(complete[SES_COLUMNS[1]]), 6)),
    "",
    "SES PC1 distribution",
    "mean: " + str(round(pc1_series.mean(), 6)),
    "std: " + str(round(pc1_series.std(), 6)),
    "min: " + str(round(pc1_series.min(), 6)),
    "25%: " + str(round(pc1_series.quantile(0.25), 6)),
    "50%: " + str(round(pc1_series.quantile(0.50), 6)),
    "75%: " + str(round(pc1_series.quantile(0.75), 6)),
    "max: " + str(round(pc1_series.max(), 6)),
]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Done:", TABLE)
print("Figure:", FIGURE)
print("Report:", REPORT)
