#!/usr/bin/env python3
"""Check whether age and grade are consistent across subject tables."""

from pathlib import Path

import pandas as pd


SUBJECT_ID = "subject_id"
REPORT = Path("/path/to/data/processed/log/age_grade_conflicts.csv")

# 每个文件第一列都作为被试编号；这里填各文件里的年龄列和年级列。
FILES = {
    # Path("/path/to/data/raw/covs/数学能力_cov.tsv"): ["入库年级"],
    # Path("/path/to/data/raw/covs/抑制控制能力_cov.tsv"): ["入库年级"],
    # Path("/path/to/data/raw/covs/中文阅读能力_cov.tsv"): ["入库年级"],
    Path("/path/to/data/processed/bhv_grade.csv"): ["grade"],
    Path("/path/to/data/raw/env/Fianl_Data_Cleaned_Z.csv"): ["grade"],
}


def get_separator(path):
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def clean_value(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.NA
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number):
        return str(round(float(number), 6))
    return str(value).strip()


def check_column_consistency(files, report):
    rows = []

    for input_file, columns in files.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        df = df.rename(columns={df.columns[0]: SUBJECT_ID})
        df[SUBJECT_ID] = df[SUBJECT_ID].astype(str).str.strip()

        one_file = df[[SUBJECT_ID] + columns].copy()
        one_file["source_file"] = str(input_file)
        rows.append(one_file)

    data = pd.concat(rows, ignore_index=True)
    conflicts = []

    for subject_id, group in data.groupby(SUBJECT_ID, sort=False):
        for column in data.columns.drop([SUBJECT_ID, "source_file"]):
            values = group[column].apply(clean_value).dropna().unique()
            if len(values) > 1:
                conflicts.append({
                    SUBJECT_ID: subject_id,
                    "column": column,
                    "values": " | ".join(values),
                    "source_files": " | ".join(group["source_file"]),
                })

    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conflicts).to_csv(report, index=False, na_rep="")
    print("Conflicts:", len(conflicts))
    print("Report:", report)


check_column_consistency(FILES, REPORT)
