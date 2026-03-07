import pandas as pd
from itertools import combinations

# 1. Load and Clean MIMIC-IV Diagnoses

mimic_path = "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"

print("Loading MIMIC-IV diagnoses data...")
diagnoses = pd.read_csv(mimic_path, dtype=str)
print(f"Loaded: {diagnoses.shape[0]:,} rows × {diagnoses.shape[1]} columns\n")

print("Columns in dataset:")
print(list(diagnoses.columns))
print("\nSample rows:")
print(diagnoses.head(10).to_string(index=False))

cols_to_keep = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
diagnoses = diagnoses[cols_to_keep]

before_filter = len(diagnoses)
diagnoses = diagnoses[diagnoses["icd_version"] == "10"]
after_filter = len(diagnoses)
print(f"ICD-10 filter applied: {before_filter:,} → {after_filter:,} rows")

before_clean = len(diagnoses)
diagnoses = diagnoses.dropna(subset=["icd_code"])
diagnoses = diagnoses[diagnoses["icd_code"].str.strip() != ""]
after_clean = len(diagnoses)
print(f"Cleaned empty ICD codes: {before_clean:,} → {after_clean:,} rows")

before_dedup = len(diagnoses)
diagnoses.drop_duplicates(subset=["subject_id", "hadm_id", "icd_code"], inplace=True)
after_dedup = len(diagnoses)
print(f"Removed duplicates: {before_dedup - after_dedup:,}")

diagnoses.reset_index(drop=True, inplace=True)

print("\n===== SUMMARY =====")
print(f"Total ICD-10 diagnoses: {len(diagnoses):,}")
print(f"Unique patients: {diagnoses['subject_id'].nunique():,}")
print(f"Unique admissions: {diagnoses['hadm_id'].nunique():,}")
print(f"Unique ICD-10 codes: {diagnoses['icd_code'].nunique():,}")

out_path = "mimic_icd10_clean.csv"
diagnoses.to_csv(out_path, index=False)
print(f"\nCleaned ICD-10 data saved → {out_path}")

print("\nSample of cleaned dataset:")
print(diagnoses.sample(10, random_state=42).to_string(index=False))

# 2. Extract ICD → SNOMED CT Mapping from Athena

print("\nLoading OHDSI Athena vocabularies...")
concepts = pd.read_csv("vocab/CONCEPT.csv", sep="\t", dtype=str, low_memory=False)
rels = pd.read_csv("vocab/CONCEPT_RELATIONSHIP.csv", sep="\t", dtype=str, low_memory=False)
print(f"Concepts: {len(concepts):,} rows, Relationships: {len(rels):,} rows")

mapping = rels[rels["relationship_id"].isin(["Maps to", "Maps to value"])]

concepts_small = concepts[["concept_id", "concept_code", "concept_name", "vocabulary_id"]]
merged = (
    mapping
    .merge(concepts_small, left_on="concept_id_1", right_on="concept_id")
    .merge(concepts_small, left_on="concept_id_2", right_on="concept_id", suffixes=("_source", "_target"))
)

icd_to_snomed = merged[
    merged["vocabulary_id_source"].isin(["ICD9CM", "ICD10CM"])
    & (merged["vocabulary_id_target"] == "SNOMED")
][[
    "concept_code_source", "concept_name_source",
    "concept_code_target", "concept_name_target", "vocabulary_id_source"
]]

icd_to_snomed.columns = [
    "ICD_CODE", "ICD_NAME", "SNOMED_ID", "SNOMED_TERM", "ICD_VERSION"
]

map_path = "icd_to_snomed_map_athena.csv"
icd_to_snomed.to_csv(map_path, index=False)
print(f"\nSaved: {map_path} ({len(icd_to_snomed):,} rows)")

print("\n===== MAPPING SUMMARY =====")
print(f"ICD-9 Concepts: {(icd_to_snomed['ICD_VERSION'] == 'ICD9CM').sum():,}")
print(f"ICD-10 Concepts: {(icd_to_snomed['ICD_VERSION'] == 'ICD10CM').sum():,}")
print(f"Unique SNOMED Concepts: {icd_to_snomed['SNOMED_ID'].nunique():,}")

print("\nSample mappings:")
print(icd_to_snomed.sample(10, random_state=42).to_string(index=False))

# 3. Build SNOMED Co-Occurrence Pairs

MIMIC_CLEAN = out_path
ATHENA_MAP = map_path
OUT_EDGES = "mimic_snomed_pairs.tsv"

print("\nLoading cleaned data and mapping...")
mimic = pd.read_csv(MIMIC_CLEAN, dtype=str)
mapping = pd.read_csv(ATHENA_MAP, dtype=str)

required_map_cols = {"ICD_CODE", "ICD_NAME", "SNOMED_ID", "SNOMED_TERM", "ICD_VERSION"}
missing = required_map_cols - set(mapping.columns)
if missing:
    raise ValueError(f"Mapping file missing required columns: {missing}")

before_version = len(mapping)
mapping = mapping[mapping["ICD_VERSION"].isin(["ICD10CM"])]
after_version = len(mapping)
print(f"Keep ICD-10 rows in mapping: {before_version:,} → {after_version:,}")

mapping["ICD_CODE"] = mapping["ICD_CODE"].str.strip().str.upper()
mimic["icd_code"] = mimic["icd_code"].str.strip().str.upper()

print("\nJoining MIMIC diagnoses to SNOMED map on ICD code...")
before_merge = len(mimic)
merged = mimic.merge(mapping, left_on="icd_code", right_on="ICD_CODE", how="inner")
after_merge = len(merged)
print(f"Joined rows: {after_merge:,} (from {before_merge:,} diagnoses)")
print(f"Unique SNOMED concepts mapped: {merged['SNOMED_ID'].nunique():,}")
print(f"Unique ICD codes mapped: {merged['icd_code'].nunique():,}")

keep_cols = ["subject_id", "hadm_id", "icd_code", "SNOMED_ID", "SNOMED_TERM"]
merged = merged[keep_cols].drop_duplicates()

print("\nBuilding co-occurrence pairs per admission...")
pairs = []
skipped_singletons = 0

for hadm_id, grp in merged.groupby("hadm_id"):
    concepts = grp[["SNOMED_ID", "SNOMED_TERM"]].drop_duplicates().values.tolist()
    if len(concepts) < 2:
        skipped_singletons += 1
        continue
    for (id_a, term_a), (id_b, term_b) in combinations(concepts, 2):
        if id_a == id_b:
            continue
        if id_a > id_b:
            id_a, id_b = id_b, id_a
            term_a, term_b = term_b, term_a
        pairs.append((id_a, term_a, "co_occurs_with", id_b, term_b))

print(f"Admissions with <2 mapped concepts (skipped): {skipped_singletons:,}")
print(f"Raw pairs generated: {len(pairs):,}")

edges_df = pd.DataFrame(pairs, columns=[
    "sourceId", "source_term", "relation_term", "destinationId", "destination_term"
])

before_dedup = len(edges_df)
edges_df.drop_duplicates(inplace=True)
after_dedup = len(edges_df)

print(f"Deduplicated edges: {before_dedup:,} → {after_dedup:,}")

edges_df.to_csv(OUT_EDGES, sep="\t", index=False)
print(f"\nSaved co-occurrence graph → {OUT_EDGES}")
print(f"Unique edges: {len(edges_df):,}")

if len(edges_df) > 0:
    print("\nSample edges:")
    print(edges_df.sample(min(10, len(edges_df)), random_state=42).to_string(index=False))
else:
    print("\nNo edges produced. Check that your ICD codes match the mapping's ICD_CODE (case and dots).")
