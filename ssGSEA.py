import matplotlib
import pandas as pd
import gseapy as gp
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np 




# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("./GDCdata/TCGA-KIRC/Transcriptome_Profiling/Gene_Expression_Quantification")
GMT_PATH   = "./HALLMARK_COMBINED.gmt"
# EXPR_COLS  = ["unstranded", "stranded_first", "stranded_second", "tpm_unstranded"]
# EXPR_COLS  = ["stranded_first", "stranded_second"]
# EXPR_COLS  = ["tpm_unstranded"]
EXPR_COLS  = ["tpm_unstranded"]

# ── 1. Load & combine all TSV files ──────────────────────────────────────────
def load_sample(fpath: Path) -> Optional[pd.Series]:    # try:
        df = pd.read_csv(fpath, sep="\t", comment="#")

        # Clean
        df = df[~df["gene_id"].str.startswith("N_")]
        df = df[df["gene_type"] == "protein_coding"]
        df = df[["gene_name"] + EXPR_COLS].copy()

        df["gene_name"] = df["gene_name"].str.strip()
        df = df.drop_duplicates(subset=["gene_name"], keep="first")
        df = df[df["gene_name"].notna() & (df["gene_name"] != "nan")]
        df = df.set_index("gene_name")

        stacked = df.stack()
        stacked.name = fpath.parent.name   # uuid directory = sample ID

        return stacked

    # except Exception as e:
    #     print(f"[WARN] Skipping {fpath.name}: {e}")
    #     return None


# Fix: glob specifically for .tsv files at depth 2
tsv_files = sorted(DATA_DIR.glob("*/*.tsv"))
print(f"Found {len(tsv_files)} TSV files")

records = []
for fpath in tqdm(tsv_files, desc="Loading samples"):
    s = load_sample(fpath)
    if s is not None:
        records.append(s)

# combined shape: rows = samples, columns = MultiIndex(gene_name, expr_col)
combined = pd.DataFrame(records)                # (n_samples × n_genes*n_cols)
combined.index.name = "sample_id"

print(f"\nCombined shape: {combined.shape}")
print(combined.iloc[:3, :6])                    # quick sanity peek


expr_matrix = (
    combined.xs("tpm_unstranded", axis=1, level=1)
    # .add(combined.xs("stranded_second", axis=1, level=1))
    # .div(2)
    .T
    .fillna(0)
)
expr_matrix.index.name = "gene_name"
print(f"\nAveraged stranded matrix for ssGSEA: {expr_matrix.shape}  (genes × samples)")

expr_matrix.to_csv("tpm_matrix.csv")

# ── 3. Run ssGSEA on all samples at once ─────────────────────────────────────
print("\nRunning ssGSEA …")
results = gp.ssgsea(
    data=expr_matrix,
    gene_sets=GMT_PATH,
    outdir=None,
    no_plot=True,
    processes=4,          
)

scores = results.res2d.pivot(index="Term", columns="Name", values="NES").T
scores.index.name = "sample_id"
print(f"\nssGSEA scores shape: {scores.shape}")
print(scores.head())


# ── 4. Label each sample ──────────────────────────────────────────────────────
def label_sample(row):
    delta = row["HALLMARK_GLYCOLYSIS"] - row["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    if   delta >  0.5: return "Glycolytic"
    elif delta < -0.5: return "Oxidative"
    else:              return "Mixed"

scores["delta"]  = scores["HALLMARK_GLYCOLYSIS"] - scores["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
scores["label"]  = scores.apply(label_sample, axis=1)

print("\nLabel distribution:")
print(scores["label"].value_counts())
print(scores[["HALLMARK_GLYCOLYSIS", "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "delta", "label"]].head(10))


# Add 4 relevant gene columns for boxplot
scores["HK2"]   = combined[("HK2", "tpm_unstranded")]
# scores["LDHA"]   = combined[("LDHA", "tpm_unstranded")]
# scores["SDHA"]   = combined[("SDHA", "tpm_unstranded")]
# scores["UQCRC1"] = combined[("UQCRC1", "tpm_unstranded")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(scores["HALLMARK_GLYCOLYSIS"], scores["HK2"], alpha=0.6)
axes[0].set_xlabel("HALLMARK_GLYCOLYSIS")
axes[0].set_ylabel("HK2")
axes[0].set_title("Glycolysis vs HK2")

axes[1].scatter(scores["HALLMARK_OXIDATIVE_PHOSPHORYLATION"], scores["HK2"], alpha=0.6)
axes[1].set_xlabel("HALLMARK_OXIDATIVE_PHOSPHORYLATION")
axes[1].set_ylabel("HK2")
axes[1].set_title("OXPHOS vs HK2")

axes[2].scatter(scores["delta"], scores["HK2"], alpha=0.6)
axes[2].set_xlabel("Delta (Glycolysis - OXPHOS)")
axes[2].set_ylabel("HK2")
axes[2].set_title("Delta vs HK2")

plt.tight_layout()
plt.savefig("hk2_scatter.png", dpi=150)
plt.show()



scores.to_csv("ssGSEA_scores_labeled.csv")


scores_subset = scores[["delta", "label"]].copy()


scores_subset.columns = pd.MultiIndex.from_product([["ssGSEA"], scores_subset.columns])

final = scores_subset.join(combined)
print(final.head(100))

final.to_csv("tcga_kirc_combined_labeled.csv")
print("\nSaved → tcga_kirc_combined_labeled.csv")

