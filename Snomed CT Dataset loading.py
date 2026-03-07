import pandas as pd

# === File paths ===
rels_path = "SnomedCT/Snapshot/Terminology/sct2_Relationship_Snapshot_US1000124_20250901.txt"
desc_path = "SnomedCT/Snapshot/Terminology/sct2_Description_Snapshot-en_US1000124_20250901.txt"

print("\n Loading SNOMED CT raw files...")
snomed_rels = pd.read_csv(rels_path, sep="\t", dtype=str)
snomed_descs = pd.read_csv(desc_path, sep="\t", dtype=str)
print(f"→ Relationships: {snomed_rels.shape}, Descriptions: {snomed_descs.shape}")

# === OPTIONAL: Keep all relations (recommended for research)
# Comment this line if you want ALL 213 relations
# snomed_rels = snomed_rels[snomed_rels["active"] == "1"]

# === Extract ALL terms (not only preferred)
desc_terms = snomed_descs[["conceptId", "term"]].dropna()
print(f"Loaded {len(desc_terms):,} description terms")

# === Build the relation-type dictionary ===
relation_ids = snomed_rels["typeId"].unique()

relation_map_df = desc_terms[desc_terms["conceptId"].isin(relation_ids)]
relation_map = dict(zip(relation_map_df["conceptId"], relation_map_df["term"]))

print(f" Found {len(relation_map)} human-readable relation names")
print(f" Relation types in raw file: {len(relation_ids)}")

# === Add readable relation names, fallback to typeId if missing
def get_relation_name(tid):
    if tid in relation_map:
        return relation_map[tid]
    return f"relation_{tid}"   # fallback to keep the typeId distinct

snomed_rels["relation_term"] = snomed_rels["typeId"].apply(get_relation_name)

# === Merge source & destination terms ===
print("\n Merging concept names...")

snomed_rels = snomed_rels.merge(
    desc_terms, left_on="sourceId", right_on="conceptId", how="left"
).rename(columns={"term": "source_term"}).drop(columns=["conceptId"])

snomed_rels = snomed_rels.merge(
    desc_terms, left_on="destinationId", right_on="conceptId", how="left"
).rename(columns={"term": "destination_term"}).drop(columns=["conceptId"])

# === Clean only rows where node names are missing
before_clean = len(snomed_rels)

snomed_rels = snomed_rels.dropna(subset=["source_term", "destination_term"])

mask_blank = (
    (snomed_rels["source_term"].str.strip() == "") |
    (snomed_rels["destination_term"].str.strip() == "")
)
snomed_rels = snomed_rels[~mask_blank]

after_clean = len(snomed_rels)
print(f" Rows removed due to missing node names: {before_clean - after_clean}")

# === Reorder columns ===
snomed_rels = snomed_rels[[
    "sourceId", "source_term",
    "relation_term", "typeId",
    "destinationId", "destination_term"
]]

# === Save output ===
out_path = "snomed_relations_full.tsv"
snomed_rels.to_csv(out_path, sep="\t", index=False)

print("\n====== SUMMARY REPORT ======")
print(f"Final dataset shape: {snomed_rels.shape}")
print(f"Unique relation types preserved: {snomed_rels['typeId'].nunique()}")
print("✨ Sample:")
print(snomed_rels.sample(5, random_state=42).to_string(index=False))
