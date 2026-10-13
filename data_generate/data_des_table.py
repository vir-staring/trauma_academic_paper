#!/usr/bin/env python3
"""Generate manuscript sample-description tables from participant intersections."""
from pathlib import Path
import pandas as pd

ENV_FILE = Path("/path/to/data/final/env_bhv_cov.csv")
DWI_FILE = Path("/path/to/data/final/dmri_net_all_harmonized_siteqc_grade.csv")
ADVERSE_FILE = Path("/path/to/data/final/psm_trauma_results/不利.csv")
NON_ADVERSE_FILE = Path("/path/to/data/final/psm_trauma_results/无不利.csv")
OUT_DIR = Path("/path/to/output/tables/sample_tables")
ID, SEX, GRADE, AGE = "subject_id", "gender", "grade", "age_month"
TRAUMA, SES, SSRT = "childhood_trauma", "f_ses", "ssrt_reaction_grade_z"
BEHAVIORS = [
    "ms_symbolic_score_grade_z", "ms_verbalized_score_grade_z",
    "ms_situational_score_grade_z", "rsw_ans_score_grade_z", "rit_corrected_fluency_grade_z",
]
TOTALS = ["cr_total_score_grade_z", "ms_total_score_grade_z"]
LABELS = {
    TRAUMA: "Childhood trauma total score", BEHAVIORS[0]: "Math problem-solving: symbolic operation",
    BEHAVIORS[1]: "Math problem-solving: conceptual understanding",
    BEHAVIORS[2]: "Math problem-solving: situational problem solving",
    BEHAVIORS[3]: "Single-character recognition", BEHAVIORS[4]: "Three-minute corrected reading fluency",
    TOTALS[0]: "Chinese reading ability total", TOTALS[1]: "Mathematics problem-solving total",
    SES: "Family SES", SSRT: "Stop-signal reaction time (SSRT)", "DWI": "DWI data",
}
STAGES = [("G1-3", 1, 3), ("G4-6", 4, 6), ("G7-9", 7, 9), ("G10-12", 10, 12)]
OVERVIEW_STAGES = [("G3", 3, 3)] + STAGES[1:]
def mean_sd(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    sd = f"{x.std():.2f}" if len(x) > 1 else "NA"
    return f"{x.mean():.2f} ({sd})"
def median_iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return "NA" if x.empty else f"{x.median():.2f} ({x.quantile(.25):.2f}, {x.quantile(.75):.2f})"
def basic(df, stages=STAGES):
    n = len(df)
    age = pd.to_numeric(df[AGE], errors="coerce").dropna()
    female = int((df[SEX] == 0).sum())
    distribution = "; ".join(
        f"{label}: {int(df[GRADE].between(low, high).sum())} ({df[GRADE].between(low, high).mean() * 100:.1f}%)"
        for label, low, high in stages
    ) if n else "NA"
    return {
        "N": n, "Age in months, M (SD)": mean_sd(age),
        "Age range (months)": "NA" if age.empty else f"{age.min():.0f}-{age.max():.0f}",
        "Female, n (%)": "NA" if not n else f"{female} ({female / n * 100:.1f}%)",
        "Family SES, M (SD)": mean_sd(df[SES]), "School stage distribution": distribution,
    }
def read_ids(path):
    data = pd.read_csv(path, dtype={ID: "string"})
    if ID not in data:
        raise ValueError(f"Missing required column in {path}: {ID}")
    data[ID] = data[ID].str.strip()
    if data[ID].isna().any() or data[ID].eq("").any() or data[ID].duplicated().any():
        raise ValueError(f"Missing, blank, or duplicate {ID} values in {path}")
    return data[[ID]]
def model_row(name, outcome, brain, stage, data, required, dwi_ids=None):
    sample = data.dropna(subset=required)
    if dwi_ids is not None:
        sample = sample[sample[ID].isin(dwi_ids)]
    return {
        "Analysis": name, "Outcome": LABELS.get(outcome, outcome), "Brain measure": brain,
        "School stage": stage, "Final N": len(sample), "Age in months, M (SD)": mean_sd(sample[AGE]),
        "Female, n (%)": basic(sample)["Female, n (%)"],
        "Childhood trauma score, M (SD)": mean_sd(sample[TRAUMA]), "SSRT, M (SD)": mean_sd(sample[SSRT]),
    }
def grouped_table(groups, outcome=None, dwi_ids=None):
    rows = []
    for stage, low, high in STAGES:
        for group in ["Adverse", "Non-adverse"]:
            part = groups[(groups["Group"] == group) & groups[GRADE].between(low, high)]
            part = part[part[outcome].notna()] if outcome else part[part[ID].isin(dwi_ids)]
            b = basic(part)
            row = {"School stage": stage, "Group": group, "N": b["N"],
                   "Age in months, M (SD)": b["Age in months, M (SD)"],
                   "Female, n (%)": b["Female, n (%)"], "Family SES, M (SD)": b["Family SES, M (SD)"],
                   "Childhood trauma score, M (SD)": mean_sd(part[TRAUMA])}
            if outcome:
                row[f"{LABELS[outcome]}, median (IQR)"] = median_iqr(part[outcome])
            rows.append(row)
    return pd.DataFrame(rows)
def main():
    env = pd.read_csv(ENV_FILE, dtype={ID: "string"})
    needed = [ID, SEX, GRADE, AGE, TRAUMA, SES, SSRT] + BEHAVIORS + TOTALS
    missing = [c for c in needed if c not in env]
    if missing:
        raise ValueError(f"Missing required columns in environment table: {missing}")
    env[ID] = env[ID].str.strip()
    if env[ID].isna().any() or env[ID].eq("").any() or env[ID].duplicated().any():
        raise ValueError(f"Missing, blank, or duplicate {ID} values in environment table")
    numeric = env[needed[1:]].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in needed[1:] if env[c].notna().any() and numeric[c].isna().sum() > env[c].isna().sum()]
    if bad:
        raise ValueError(f"Non-numeric values found in numeric columns: {bad}")
    env[needed[1:]] = numeric
    valid_grade = env[GRADE].dropna()
    if (not env[SEX].dropna().isin([0, 1]).all() or not valid_grade.between(1, 12).all()
            or not valid_grade.mod(1).eq(0).all()):
        raise ValueError("gender must be coded 0/1 and grade must be an integer from 1 to 12")
    dwi_ids = set(read_ids(DWI_FILE)[ID])
    adverse, non_adverse = read_ids(ADVERSE_FILE), read_ids(NON_ADVERSE_FILE)
    if set(adverse[ID]) & set(non_adverse[ID]):
        raise ValueError("The adverse and non-adverse group files contain overlapping participants")
    groups = pd.concat([adverse.assign(Group="Adverse"), non_adverse.assign(Group="Non-adverse")])
    unknown = set(groups[ID]) - set(env[ID])
    if unknown:
        raise ValueError(f"{len(unknown)} participants in the group files are absent from the environment table")
    groups = groups.merge(env, on=ID, validate="one_to_one")
    if (groups[TRAUMA].isna().any() or (groups.loc[groups["Group"] == "Adverse", TRAUMA] <= 2).any()
            or (groups.loc[groups["Group"] == "Non-adverse", TRAUMA] > 2).any()):
        raise ValueError("Group files are inconsistent with the childhood_trauma > 2 threshold")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    overview = env[env[GRADE].between(3, 12)]
    rows = [{"Variable": f"{LABELS[v]} ({v})", **basic(overview[overview[v].notna()], OVERVIEW_STAGES)}
            for v in [TRAUMA] + BEHAVIORS + TOTALS + [SES, SSRT]]
    rows.append({"Variable": "DWI data", **basic(overview[overview[ID].isin(dwi_ids)], OVERVIEW_STAGES)})
    outputs["table1_variable_samples.csv"] = pd.DataFrame(rows)
    for letter, outcome in zip("ABCDE", BEHAVIORS):
        outputs[f"table2{letter}_{outcome}.csv"] = grouped_table(groups, outcome=outcome)
    outputs["table2F_DWI.csv"] = grouped_table(groups, dwi_ids=dwi_ids)
    table_specs = [
        ("table3_mediation_samples.csv", False, True),
        ("table4A_moderation_samples.csv", True, False),
        ("table4B_triple_moderation_samples.csv", True, True),
    ]
    for filename, use_ssrt, use_dwi in table_specs:
        rows = []
        for outcome in TOTALS:
            required = [TRAUMA, outcome, SEX, SES] + ([SSRT] if use_ssrt else [])
            part = env.dropna(subset=required)
            part = part[part[ID].isin(dwi_ids)] if use_dwi else part
            b = basic(part)
            row = {"Outcome": LABELS[outcome], "N": b["N"], "Age in months, M (SD)": b["Age in months, M (SD)"],
                   "Female, n (%)": b["Female, n (%)"], "Family SES, M (SD)": b["Family SES, M (SD)"],
                   "Childhood trauma score, M (SD)": mean_sd(part[TRAUMA]),
                   "Outcome score, median (IQR)": median_iqr(part[outcome])}
            if use_ssrt: row["SSRT, M (SD)"] = mean_sd(part[SSRT])
            if use_dwi: row["Brain data"] = "DWI"
            rows.append(row)
        outputs[filename] = pd.DataFrame(rows)
    model_rows = []
    for outcome in BEHAVIORS:
        model_rows.append(model_row("Trauma-behavior partial correlation", outcome, "None", "Full sample", env, [TRAUMA, outcome, SEX, SES]))
        for stage, low, high in STAGES:
            part = groups[groups[GRADE].between(low, high)]
            model_rows.append(model_row("Stage-specific trauma group comparison", outcome, "None", stage, part, [outcome]))
    model_rows.append(model_row("Trauma-DWI partial correlation", "DWI", "DWI", "Full sample", env, [TRAUMA, SEX, SES], dwi_ids))
    model_rows.append(model_row("Trauma group GAM", "DWI", "DWI", "Full sample", groups, [TRAUMA, AGE], dwi_ids))
    for outcome in TOTALS:
        model_rows.append(model_row("Simple mediation", outcome, "DWI", "Full sample", env, [TRAUMA, outcome, SEX, SES], dwi_ids))
        model_rows.append(model_row("Age-moderated mediation", outcome, "DWI", "Full sample", env, [TRAUMA, outcome, AGE, SEX, SES], dwi_ids))
        model_rows.append(model_row("Two-way moderation", outcome, "None", "Full sample", env, [TRAUMA, outcome, SSRT, SEX, SES]))
        model_rows.append(model_row("Three-way moderation", outcome, "DWI", "Full sample", env, [TRAUMA, outcome, SSRT, SEX, SES], dwi_ids))
    outputs["table5_analysis_sample_summary.csv"] = pd.DataFrame(model_rows)
    for name, table in outputs.items():
        path = OUT_DIR / name
        table.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")
        check = pd.read_csv(path)
        if list(check.columns) != list(table.columns) or len(check) != len(table):
            raise RuntimeError(f"Output validation failed: {path}")
    print(f"Done: {len(outputs)} tables saved to {OUT_DIR}")
if __name__ == "__main__":
    main()
