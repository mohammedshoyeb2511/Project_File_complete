import pandas as pd
from collections import Counter

# Step 1 — Load dataset

MERGED_PATH = "merged_relations.tsv"
print(f"Loading merged dataset: {MERGED_PATH}")
df = pd.read_csv(MERGED_PATH, sep="\t", dtype=str)
print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns\n")

required_cols = ["sourceId", "source_term", "relation_term", "destinationId", "destination_term"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Step 2 — Drop null or blank

before = len(df)
df.dropna(subset=["source_term", "destination_term"], inplace=True)
df = df[(df["source_term"].str.strip() != "") & (df["destination_term"].str.strip() != "")]
after = len(df)
print(f"Removed empty terms: {before:,} → {after:,}")

# Step 3 — Exact duplicate removal

before = len(df)
df.drop_duplicates(subset=["sourceId", "relation_term", "destinationId"], inplace=True)
after = len(df)
print(f"Removed {before - after:,} exact duplicate edges")

# Step 4 — Safe symmetric deduplication

symmetric_rels = {"associated_with", "co_occurs_with"}
sym_df = df[df["relation_term"].isin(symmetric_rels)].copy()
sym_df["pair_key"] = sym_df.apply(
    lambda x: "_".join(sorted([x["sourceId"], x["destinationId"]])), axis=1
)
before = len(sym_df)
sym_df = sym_df.drop_duplicates(subset=["pair_key", "relation_term"])
after = len(sym_df)
print(f"Removed {before - after:,} symmetric duplicates")

nonsym_df = df[~df["relation_term"].isin(symmetric_rels)]
df = pd.concat([nonsym_df, sym_df.drop(columns=["pair_key"])])
print(f"After symmetric cleanup: {len(df):,} total edges")

# Step 5 — Detect conflicts (same pair, multiple relations)
conflicts = (
    df.groupby(["sourceId", "destinationId"])["relation_term"].nunique().reset_index()
)
conflicts = conflicts[conflicts["relation_term"] > 1]
print(f"Conflicting edges: {len(conflicts):,}")
if len(conflicts) > 0:
    conflict_edges = df.merge(
        conflicts[["sourceId", "destinationId"]],
        on=["sourceId", "destinationId"], how="inner"
    )
    conflict_edges.to_csv("conflicting_relations.tsv", sep="\t", index=False)
    print("Saved: conflicting_relations.tsv")

# Step 6 — Context-aware hub filtering
print("\nAnalyzing node degrees...")
all_nodes = pd.concat([df["source_term"], df["destination_term"]])
node_freq = Counter(all_nodes)
hub_nodes = [n for n, f in node_freq.items() if f > 200]  # threshold adjustable
print(f"Found {len(hub_nodes)} hub nodes (>200 edges)")

before = len(df)
mask = (
    (df["relation_term"] == "co_occurs_with") &
    (df["source_term"].isin(hub_nodes) | df["destination_term"].isin(hub_nodes))
)
removed = df[mask]
df = df[~mask]
after = len(df)
print(f"Removed {before - after:,} noisy co-occurrence edges from hub nodes")

if len(removed) > 0:
    removed.to_csv("removed_hub_edges.tsv", sep="\t", index=False)
    print("Saved: removed_hub_edges.tsv")

# Step 7 — Remove generic / non-informative terms
generic_keywords = ["disease", "disorder", "finding", "symptom", "clinical", "observation"]
before = len(df)
mask = (
    df["source_term"].str.lower().apply(lambda x: any(k in x for k in generic_keywords)) |
    df["destination_term"].str.lower().apply(lambda x: any(k in x for k in generic_keywords))
)
df = df[~mask]
after = len(df)
print(f"Removed {before - after:,} generic concept edges")

# Step 8 — Save and summarize
CLEAN_PATH = "Final_Snomed_CT_and_MIMIC-IV_dataset.tsv"
df.to_csv(CLEAN_PATH, sep="\t", index=False)
print(f"\nSaved clean dataset: {CLEAN_PATH}")
print(f"Final edges: {len(df):,}")
print(f"Unique nodes: {len(set(pd.concat([df['sourceId'], df['destinationId']]))) :,}")

print("\nSample edges:")
print(df.sample(10, random_state=42).to_string(index=False))
