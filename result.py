import pandas as pd

# ----------------------------
# 1) CONFIG
# ----------------------------
DIAG_PATH = "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"  # update if needed
TARGET_CODES = ["F05", "I10", "J90", "G92"]

OUT_FILTERED = "mimic_icd10_F05_I10_J90_G92_rows.csv"
OUT_ADM_LEVEL = "mimic_icd10_F05_I10_J90_G92_by_admission.csv"


# ----------------------------
# 2) LOAD DATA (IMPORTANT: comma-separated)
# ----------------------------
df = pd.read_csv(
    DIAG_PATH,
    compression="gzip"   # sep defaults to comma (correct for this file)
)

print("Loaded diagnoses_icd")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------
# 3) CLEAN / TYPE SAFETY
# ----------------------------
# Ensure expected columns exist
expected_cols = {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Your file may be read with the wrong separator. "
        f"Columns seen: {list(df.columns)}"
    )

# Make sure icd_version is numeric (sometimes reads as object/string)
df["icd_version"] = pd.to_numeric(df["icd_version"], errors="coerce")

# Normalize ICD codes: strip spaces, uppercase (safe)
df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()

# ----------------------------
# 4) FILTER ICD-10 + TARGET CODES
# ----------------------------
df_filtered = df[
    (df["icd_version"] == 10) &
    (df["icd_code"].isin(TARGET_CODES))
].copy()

print("\nFiltered ICD-10 rows for:", TARGET_CODES)
print("Filtered shape:", df_filtered.shape)

# Show a quick sample
print("\nSample filtered rows:")
print(df_filtered.head(10).to_string(index=False))

# Counts per code
print("\nCounts per ICD-10 code:")
print(df_filtered["icd_code"].value_counts(dropna=False).to_string())

# ----------------------------
# 5) OPTIONAL: PRIMARY DIAGNOSES ONLY
# ----------------------------
df_primary = df_filtered[df_filtered["seq_num"] == 1].copy()
print("\nPrimary diagnosis only (seq_num == 1) shape:", df_primary.shape)

# ----------------------------
# 6) ADMISSION-LEVEL COHORT TABLE
#    one row per (subject_id, hadm_id) with list of matched codes
# ----------------------------
df_by_admission = (
    df_filtered
    .groupby(["subject_id", "hadm_id"])["icd_code"]
    .apply(lambda x: sorted(set(x.tolist())))
    .reset_index()
)

# Make it easier to read/save: join list into a string
df_by_admission["matched_icd10_codes"] = df_by_admission["icd_code"].apply(lambda xs: ",".join(xs))
df_by_admission = df_by_admission.drop(columns=["icd_code"])

print("\nAdmission-level cohort table shape:", df_by_admission.shape)
print("\nSample admission-level rows:")
print(df_by_admission.head(10).to_string(index=False))

# ----------------------------
# 7) SAVE OUTPUTS
# ----------------------------
df_filtered.to_csv(OUT_FILTERED, index=False)
df_by_admission.to_csv(OUT_ADM_LEVEL, index=False)

print("\nSaved:")
print(" -", OUT_FILTERED)
print(" -", OUT_ADM_LEVEL)
