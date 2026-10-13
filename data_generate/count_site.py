#!/usr/bin/env python3
"""Count unique subjects by enrollment site across TSV tables."""

from pathlib import Path

import pandas as pd


# 在这里填写需要统计的 TSV 文件路径。
TSV_FILES = [
    # Path("/path/to/data/横断队列被试信息表.tsv"),
    # Path("/path/to/data/横断队列被试信息表.tsv"),
    # Path("/path/to/data/横断队列被试信息表.tsv"),
    # Path("/path/to/data/横断队列被试信息表.tsv"),
    # Path("/path/to/data/横断队列被试信息表.tsv"),
    Path("/path/to/data/横断队列被试信息表.tsv"),

]

SITE_COLUMN = "入库站点"


def count_subjects_by_site(tsv_files, site_column):
    if not tsv_files:
        raise ValueError("TSV_FILES 为空，请先填写至少一个 TSV 文件路径")

    subject_sites = {}

    for path in map(Path, tsv_files):
        data = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if data.shape[1] == 0:
            raise ValueError(f"表格没有列：{path}")
        if site_column not in data.columns:
            raise ValueError(f"表格缺少站点列 {site_column!r}：{path}")

        subject_column = data.columns[0]
        selected = data[[subject_column, site_column]].copy()
        selected[subject_column] = selected[subject_column].str.strip()
        selected[site_column] = selected[site_column].str.strip()

        if selected[subject_column].eq("").any():
            raise ValueError(f"被试编号列存在空值：{path}")
        if selected[site_column].eq("").any():
            raise ValueError(f"站点列存在空值：{path}")

        for subject_id, site in selected.itertuples(index=False, name=None):
            previous_site = subject_sites.get(subject_id)
            if previous_site is not None and previous_site != site:
                raise ValueError(
                    f"被试 {subject_id!r} 同时对应多个站点："
                    f"{previous_site!r} 和 {site!r}"
                )
            subject_sites[subject_id] = site

    counts = pd.Series(subject_sites.values()).value_counts().sort_index()
    return counts


def main():
    counts = count_subjects_by_site(TSV_FILES, SITE_COLUMN)
    print(f"不同入库站点数：{len(counts)}")
    print("各站点被试数：")
    for site, count in counts.items():
        print(f"{site}\t{count}")


if __name__ == "__main__":
    main()
