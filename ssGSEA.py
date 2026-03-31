import pandas as pd
import gseapy as gp
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./GDCdata/TCGA-KIRC/Transcriptome_Profiling/Gene_Expression_Quantification")
GMT_PATH = "./HALLMARK_COMBINED.gmt"
EXPR_COLS = ["tpm_unstranded"]


# Clean and load a single sample TSV file
def load_sample(fpath: Path) -> Optional[pd.Series]:
    df = pd.read_csv(fpath, sep="\t", comment="#")

    df = df[~df["gene_id"].str.startswith("N_")]
    df = df[df["gene_type"] == "protein_coding"]
    df = df[["gene_name"] + EXPR_COLS].copy()

    df["gene_name"] = df["gene_name"].str.strip()
    df = df.drop_duplicates(subset="gene_name")
    df = df[df["gene_name"].notna() & (df["gene_name"] != "nan")]

    df = df.set_index("gene_name")

    stacked = df.stack()
    stacked.name = fpath.parent.name  # sample ID

    return stacked

# Load all TSV files into a combined DataFrame
def load_all_samples(data_dir: Path) -> pd.DataFrame:
    tsv_files = sorted(data_dir.glob("*/*.tsv"))
    print(f"Found {len(tsv_files)} TSV files")

    records = [
        s for f in tqdm(tsv_files, desc="Loading samples")
        if (s := load_sample(f)) is not None
    ]

    combined = pd.DataFrame(records)
    combined.index.name = "sample_id"
    return combined


# Extracts the desired fields and builds an expression matrix 
def build_expression_matrix(combined: pd.DataFrame) -> pd.DataFrame:
    """Convert combined data into (genes × samples) matrix."""
    expr = (
        combined.xs("tpm_unstranded", axis=1, level=1)
        .T
        .fillna(0)
    )
    expr.index.name = "gene_name"
    return expr


# Runs ssGSEA and returns a DataFrame of pathway activity scores 
def run_ssgsea(expr_matrix: pd.DataFrame, gmt_path: str) -> pd.DataFrame:
    print("Running ssGSEA...")

    results = gp.ssgsea(
        data=expr_matrix,
        gene_sets=gmt_path,
        outdir=None,
        no_plot=True,
        threads=12,
    )

    scores = results.res2d.pivot(index="Term", columns="Name", values="NES").T
    scores.index.name = "sample_id"
    scores.columns.name = "pathway"

    return scores


# Normalize scores among glycolysis and OXPHOS pathways
def normalize_scores(scores: pd.DataFrame, pathways: List[str]) -> pd.DataFrame:
    """Z-score normalize selected pathways."""
    for col in pathways:
        scores[col] = (scores[col] - scores[col].mean()) / scores[col].std()
    return scores

# Compute delta and assign metabolic labels based on delta thresholds
def label_samples(scores: pd.DataFrame) -> pd.DataFrame:
    scores["delta"] = (
        scores["HALLMARK_GLYCOLYSIS"] -
        scores["HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    )

    def classify(delta):
        if delta > 0.5:
            return "Glycolytic"
        elif delta < -0.5:
            return "Oxidative"
        return "Mixed"

    scores["label"] = scores["delta"].apply(classify)
    return scores


# Plot distribution of sample labels
def plot_label_distribution(scores: pd.DataFrame):
    label_counts = scores["label"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index,
            autopct='%1.1f%%', startangle=90)
    plt.title("Sample Label Distribution (ssGSEA)")
    plt.axis("equal")
    plt.savefig("label_distribution_pie.png", dpi=150)

# Addes HK2 (Glycolysis correlative) and UQCRC1 (OXPHOS correlative) expression values to the scores DataFrame for scatter plotting
def add_gene_expression(scores: pd.DataFrame, combined: pd.DataFrame):
    for gene in ["HK2", "UQCRC1"]:
        scores[gene] = combined[(gene, "tpm_unstranded")]


# Plots scatter plots of pathway scores vs gene expression for HK2(Glycolysis correlative) and UQCRC1(OXPHOS correlative)
def plot_scatter(scores: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(25, 5))

    pairs = [
        ("HALLMARK_GLYCOLYSIS", "HK2"),
        ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HK2"),
        ("delta", "HK2"),
        ("HALLMARK_GLYCOLYSIS", "UQCRC1"),
        ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "UQCRC1"),
        ("delta", "UQCRC1"),
    ]

    titles = [
        "Glycolysis vs HK2",
        "OXPHOS vs HK2",
        "Delta vs HK2",
        "Glycolysis vs UQCRC1",
        "OXPHOS vs UQCRC1",
        "Delta vs UQCRC1",
    ]

    for ax, (x, y), title in zip(axes, pairs, titles):
        ax.scatter(scores[x], scores[y], alpha=0.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("hk2_uqcrc1_scatter.png", dpi=150)


# Plots histogram of delta scores with a KDE curves
def plot_delta_distribution(scores: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.histplot(scores["delta"], bins=50, kde=True)
    plt.xlabel("Delta (Glycolysis - OXPHOS)")
    plt.ylabel("Count")
    plt.title("Delta Distribution with Density Curve")
    plt.grid(True)
    plt.savefig("delta_histogram_kde.png", dpi=150)
    plt.show()


# Saves outputs into readable CSV files for downstream analysis and visualization
def save_outputs(scores: pd.DataFrame, combined: pd.DataFrame):
    scores.to_csv("ssGSEA_scores_labeled.csv")

    subset = scores[["delta", "label"]].copy()
    subset.columns = pd.MultiIndex.from_product([["ssGSEA"], subset.columns])

    final = subset.join(combined)
    final.to_csv("tcga_kirc_combined_labeled.csv")


# Main
def main():
    # Load and preprocess data to build expression matrix
    combined = load_all_samples(DATA_DIR)
    expr_matrix = build_expression_matrix(combined)
    expr_matrix.to_csv("tpm_matrix.csv")

    # Run ssGSEA and process scores
    scores = run_ssgsea(expr_matrix, GMT_PATH)
    scores = normalize_scores(scores, ["HALLMARK_GLYCOLYSIS", "HALLMARK_OXIDATIVE_PHOSPHORYLATION"])
    scores = label_samples(scores)
    print(scores["label"].value_counts())

    # Plot distribution of sample labels into a pie chart
    plot_label_distribution(scores)

    # Plot scatter plots of pathway scores vs gene expression for HK2(Glycolysis correlative) and UQCRC1(OXPHOS correlative)
    add_gene_expression(scores, combined)
    plot_scatter(scores)

    # Plot histogram of delta scores with a KDE curves
    plot_delta_distribution(scores)

    #Save outputs into readable CSV files for downstream analysis 
    save_outputs(scores, combined)


if __name__ == "__main__":
    main()