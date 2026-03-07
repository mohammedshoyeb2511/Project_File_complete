import pandas as pd

SNOMED_PATH = "snomed_relations_full.tsv"   
MIMIC_PATH  = "mimic_snomed_pairs.tsv"      
OUT_PATH    = "merged_relations.tsv"

print("Loading input files...")
snomed_df = pd.read_csv(SNOMED_PATH, sep="\t", dtype=str)
mimic_df  = pd.read_csv(MIMIC_PATH, sep="\t", dtype=str)
print(f"SNOMED relations: {len(snomed_df):,}")
print(f"MIMIC co-occurrences: {len(mimic_df):,}\n")

# === Normalize columns for consistency ===
required_cols = ["sourceId", "source_term", "relation_term", "destinationId", "destination_term"]

def ensure_cols(df, name):
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return df[required_cols].copy()

snomed_df = ensure_cols(snomed_df, "SNOMED")
mimic_df  = ensure_cols(mimic_df, "MIMIC")

# === Add dataset source label (optional) ===
snomed_df["source"] = "SNOMED_CT"
mimic_df["source"]  = "MIMIC_IV"

# === Merge ===
print("Merging datasets...")
before_merge_snomed = len(snomed_df)
before_merge_mimic  = len(mimic_df)
merged = pd.concat([snomed_df, mimic_df], ignore_index=True)

after_merge = len(merged)
print(f"Before merge: SNOMED={before_merge_snomed:,}, MIMIC={before_merge_mimic:,}")
print(f"After merge (combined): {after_merge:,}")

# === Remove duplicates ===
before_clean = len(merged)
merged.drop_duplicates(subset=["sourceId", "relation_term", "destinationId"], inplace=True)
after_clean = len(merged)
print(f"Duplicate removal: {before_clean:,} → {after_clean:,} (removed {before_clean - after_clean:,})")

# === Summary ===
print("\n===== SUMMARY =====")
print("Relation type distribution:")
print(merged["relation_term"].value_counts().to_string())

print("\nSource breakdown:")
print(merged["source"].value_counts().to_string())

print(f"\nSaving merged graph → {OUT_PATH}")
merged.to_csv(OUT_PATH, sep="\t", index=False)
print(f"Merged relational graph saved: {len(merged):,} rows")

# === Optional preview ===
print("\nSample edges:")
print(merged.sample(min(10, len(merged)), random_state=42).to_string(index=False))
