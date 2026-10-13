#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


def get_separator(path):
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def extract_and_rename_columns(input_file, output_file, column_rename):
    # 根据文件后缀自动选择 CSV 或 TSV 的分隔符。
    input_sep = get_separator(input_file)
    output_sep = get_separator(output_file)
    df = pd.read_csv(input_file, sep=input_sep)

    # 按 COLUMN_RENAME 的顺序提取列，并同时改成新列名。
    selected = df[list(column_rename)].rename(columns=column_rename)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_file, sep=output_sep, index=False, na_rep="")

    print("Done:", output_file)


def merge_selected_columns(files_columns, output_file, subject_id):
    tables = []

    for input_file, selected_columns in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        df = df.rename(columns={df.columns[0]: subject_id})
        df[subject_id] = df[subject_id].astype(str).str.strip()

        if selected_columns:
            df = df[[subject_id] + selected_columns]

        one_file = df.drop_duplicates(subject_id)
        tables.append(one_file)

    # 按被试编号合并：同名列自动补成一列，不同列名正常保留。
    merged = pd.concat(tables, ignore_index=True)
    merged = merged.groupby(subject_id, as_index=False, sort=False).first()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, sep=get_separator(output_file), index=False, na_rep="")
    print("Done:", output_file)


def add_age_month_column(files_columns):
    for input_file, age_column in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))

        # 把“X岁X个月X天”转成月龄；天数按 30 天折算为 1 个月。
        age_parts = df[age_column].astype("string").str.extract(r"(\d+)岁(\d+)个月(\d+)天")
        years = pd.to_numeric(age_parts[0])
        months = pd.to_numeric(age_parts[1])
        days = pd.to_numeric(age_parts[2])
        df["采集月龄"] = (years * 12 + months + days / 30).round(2)
        df.to_csv(input_file, sep=get_separator(input_file), index=False, na_rep="")
        print("Done:", input_file)


def recode_grade_column(files_columns):
    grade_map = {
        "一年级": 1,
        "二年级": 2,
        "三年级": 3,
        "四年级": 4,
        "五年级": 5,
        "六年级": 6,
        "七年级": 7,
        "八年级": 8,
        "九年级": 9,
        "高一": 10,
        "高二": 11,
        "高三": 12,
    }

    for input_file, grade_column in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        df[grade_column] = df[grade_column].astype("string").str.strip().map(grade_map)
        df.to_csv(input_file, sep=get_separator(input_file), index=False, na_rep="")
        print("Done:", input_file)

def add_grade_z_columns(files_columns):
    for input_file, info in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        grade_column = info["grade"]

        for column in info["columns"]:
            values = pd.to_numeric(df[column], errors="coerce")
            means = values.groupby(df[grade_column]).transform("mean")
            stds = values.groupby(df[grade_column]).transform("std")
            df[column + "_grade_z"] = ((values - means) / stds).round(6)

        df.to_csv(input_file, sep=get_separator(input_file), index=False, na_rep="")
        print("Done:", input_file)

def add_two_column_sum(files_columns):
    for input_file, info in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        values = df[info["columns"]].apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1)
        df[info["new_column"]] = pd.NA
        df.loc[complete, info["new_column"]] = values.loc[complete].sum(axis=1)
        df.to_csv(input_file, sep=get_separator(input_file), index=False, na_rep="")
        print("Done:", input_file)

def split_by_school_stage(files_columns, output_dir, subject_id):
    stages = {
        "1": [1, 2, 3],
        "2": [4, 5, 6],
        "3": [7, 8, 9],
        "4": [10, 11, 12],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file, grade_column in files_columns.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        df = df.rename(columns={df.columns[0]: subject_id})
        df[subject_id] = df[subject_id].astype(str).str.strip()
        grades = pd.to_numeric(df[grade_column], errors="coerce")

        for stage_name, grade_values in stages.items():
            out = output_dir / (input_file.stem + "_" + stage_name + input_file.suffix)
            df[grades.isin(grade_values)].to_csv(out, sep=get_separator(out), index=False, na_rep="")
            print("Done:", out)

def save_complete_cases(tasks, subject_id):
    for input_file, info in tasks.items():
        df = pd.read_csv(input_file, sep=get_separator(input_file))
        df = df.rename(columns={df.columns[0]: subject_id})
        df[subject_id] = df[subject_id].astype(str).str.strip()

        # 保存“被试编号 + 指定列”，并删除这些列里有空值的行。
        out = df[[subject_id] + info["columns"]].dropna()
        output_file = info["output"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, sep=get_separator(output_file), index=False, na_rep="")
        print("Done:", output_file)

# 如果只需要处理单个文件并重命名列，可以改用下面这一行：
# INPUT_FILE = Path("/path/to/data/raw/env/Fianl_Data_Cleaned_Z.csv")
# OUTPUT_FILE = Path("/path/to/data/processed/envs.csv")
# SUBJECT_ID = "subject_id"

# COLUMN_RENAME = {
#     #envs
#     "userid_s": "subject_id",
#     "gender": "gender",
#     "grade": "grade",
#     "age_month":"age_month",
#     "si2_tnqcs_mean":"childhood_trauma",
#     "si2_tnqcs_hsmean":"childhood_abuse",
#     "si2_tnqcs_ndmean":"childhood_neglect",
#     "pf1_fmzy_max":"Parent occupation",
#     "pf1_fmsjysp_max":"Parent education",

# }
# extract_and_rename_columns(INPUT_FILE, OUTPUT_FILE, COLUMN_RENAME)

# 如果需要合并多个文件的指定列，可以改用下面这一行：
# MERGE_OUTPUT_FILE = Path("/path/to/data/processed/bhv.csv")
# SUBJECT_ID = "subject_id"

# MERGE_FILES_COLUMNS = {
#     # Path("/path/to/data/raw/bhv/数学能力.tsv"): ["数学问题解决_数学运算子维度得分", "数学问题解决_概念理解子维度得分", "数学问题解决_应用题解决子维度得分", "数学问题解决_总得分"],
#     # Path("/path/to/data/raw/bhv/抑制控制能力.tsv"): ["停止信号_停止信号反应时(ms)"],
#     # Path("/path/to/data/raw/bhv/中文阅读能力.tsv"): ["单字认读_作答得分", "单字认读_任务得分", "三分钟快速阅读_矫正阅读流畅性"],

#     Path("/path/to/data/raw/covs/数学能力_cov.tsv"): "采集年龄",
#     Path("/path/to/data/raw/covs/抑制控制能力_cov.tsv"): "采集年龄",
#     Path("/path/to/data/raw/covs/中文阅读能力_cov.tsv"): "采集年龄",
# }
# merge_selected_columns(MERGE_FILES_COLUMNS, MERGE_OUTPUT_FILE, SUBJECT_ID)

# # 每个文件对应自己的采集年龄列名。
# AGE_FILES_COLUMNS = {
#     Path("/path/to/data/raw/covs/数学能力_cov.tsv"): "采集年龄",
#     Path("/path/to/data/raw/covs/抑制控制能力_cov.tsv"): "采集年龄",
#     Path("/path/to/data/raw/covs/中文阅读能力_cov.tsv"): "采集年龄",
# }
# add_age_month_column(AGE_FILES_COLUMNS)

# # 
# GRADE_FILES_COLUMNS = {
#     # Path("/path/to/data/raw/covs/数学能力_cov.tsv"): "入库年级",
#     # Path("/path/to/data/raw/covs/抑制控制能力_cov.tsv"): "入库年级",
#     # Path("/path/to/data/raw/covs/中文阅读能力_cov.tsv"): "入库年级",
#     # Path("/path/to/data/raw/covs/brain_covs.tsv"): "入库年级",
#     Path("/path/to/data/processed/bhv_grade.csv"): "入库年级",
# }
# recode_grade_column(GRADE_FILES_COLUMNS)

# MERGE_OUTPUT_FILE = Path("/path/to/data/final/env_bhv_cov.csv")
# SUBJECT_ID = "subject_id"

# MERGE_FILES_COLUMNS = {
#     # Path("/path/to/data/raw/bhv/数学能力.tsv"): ["ms_symbolic_score", "ms_verbalized_score", "ms_situational_score", "ms_total_score"],
#     # Path("/path/to/data/raw/bhv/抑制控制能力.tsv"): ["ssrt_reaction"],
#     # Path("/path/to/data/raw/bhv/中文阅读能力.tsv"): ["rsw_ans_score", "rit_corrected_fluency"],
#     # Path("/path/to/data/processed/bhv_grade.csv"): ["grade"],

#     # Path("/path/to/data/raw/covs/数学能力_cov.tsv"): ["入库年级"],
#     # Path("/path/to/data/raw/covs/抑制控制能力_cov.tsv"): ["入库年级"],
#     # Path("/path/to/data/raw/covs/中文阅读能力_cov.tsv"): ["入库年级"],
#     Path("/path/to/data/processed/envs.csv"): [],
#     Path("/path/to/data/processed/bhvwithgrade.csv"): [],
    
# }
# merge_selected_columns(MERGE_FILES_COLUMNS, MERGE_OUTPUT_FILE, SUBJECT_ID)

# GRADE_Z_FILES = {
#     Path("/path/to/data/processed/bhvwithgrade.csv"): {
#         "grade": "grade",
#         "columns": ["ms_symbolic_score", "ms_verbalized_score", "ms_situational_score", "ms_total_score", "rit_corrected_fluency", "rsw_ans_score", "ssrt_reaction"],
#     },
#         Path("/path/to/data/processed/dmri_net_all_harmonized_siteqc_grade.csv"): {
#         "grade": "grade",
#         "columns": [    
#             "Visual1_FA",
#             "Visual2_FA",
#             "Auditory_FA",
#             "Somatomotor_FA",
#             "Orbito-Affective_FA",
#             "Ventral-Multimodal_FA",
#             "Posterior-Multimodal_FA",
#             "Cingulo-Opercular_FA",
#             "Default_FA",
#             "Language_FA",
#             "Frontoparietal_FA",
#             "Dorsal-Attention_FA"
#         ],
#     },
# }

# add_grade_z_columns(GRADE_Z_FILES)

# SUM_FILES_COLUMNS = {
#     Path("/path/to/data/final/env_bhv_cov.csv"): {
#         "columns": ["rit_corrected_fluency_grade_z", "rsw_ans_score_grade_z"],
#         "new_column": "cr_total_score_grade_z",
#     },
# }
# add_two_column_sum(SUM_FILES_COLUMNS)

# SUBJECT_ID = "subject_id"
# STAGE_OUTPUT_DIR = Path("/path/to/data/final/stage_split")
# STAGE_SPLIT_FILES = {
#     Path("/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv"): "grade",
#     Path("/path/to/data/final/env_bhv_cov.csv"): "grade",

# }

# split_by_school_stage(STAGE_SPLIT_FILES, STAGE_OUTPUT_DIR, SUBJECT_ID)

SUBJECT_ID = "subject_id"
COMPLETE_CASE_TASKS = {
    Path("/path/to/data/final/env_bhv_cov.csv"): {
        "columns": [
            # "childhood_trauma",  
            # "ms_total_score_grade_z",
            # "cr_total_score_grade_z",
            # "ssrt_reaction_grade_z",
            "age_month",
            # "f_ses",
            "grade",
            # "Parent occupation",
            # "Parent education",
            "gender",
            ],
        "output": Path("/path/to/data/final/match_covs.csv"),
    },
}
save_complete_cases(COMPLETE_CASE_TASKS, SUBJECT_ID)

# SUBJECT_ID = "subject_id"
# COMPLETE_CASE_TASKS = {
#     Path("/path/to/data/final/trauma_covs.csv"): {
#         "columns": [
#             "childhood_trauma",  
#             # "ms_total_score_grade_z",
#             # "cr_total_score_grade_z",
#             # "ssrt_reaction_grade_z",
#             # "age_month",
#             # "f_ses",
#             # "gender",
#             ],
#         "output": Path("/path/to/data/final/trauma.csv"),
#     },
# }
# save_complete_cases(COMPLETE_CASE_TASKS, SUBJECT_ID)
