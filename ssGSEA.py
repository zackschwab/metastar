import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


# Config
DATA_DIR = Path("./GDCdata/TCGA-KIRC/Transcriptome_Profiling/Gene_Expression_Quantification")
GMT_PATH = "./HALLMARK_COMBINED.gmt"
EXPR_COLS = ["tpm_unstranded"]


# Clean and load a single sample TSV file
def load_sample(fpath: Path) -> Optional[pd.Series]:

    df = pd.read_csv(fpath, sep="\t", comment="#")

    df = df[~df["gene_id"].str.startswith("N_")]
    # df = df[df["gene_type"] == "protein_coding"]
    df = df[["gene_name"] + EXPR_COLS].copy()

    df["gene_name"] = df["gene_name"].str.strip()
    df = df.drop_duplicates(subset="gene_name")
    df = df[df["gene_name"].notna() & (df["gene_name"] != "nan")]

    df = df.set_index("gene_name")

    stacked = df.stack()
    stacked.name = fpath.parent.name

    return stacked


# Load all TSV files
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


# Build expression matrix
def build_expression_matrix(combined: pd.DataFrame) -> pd.DataFrame:

    expr = (
        combined.xs("tpm_unstranded", axis=1, level=1)
        .T
        .fillna(0)
    )

    expr.index.name = "gene_name"

    return expr


# Run ssGSEA
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


# Normalize pathway scores
def normalize_scores(scores: pd.DataFrame, pathways: List[str]) -> pd.DataFrame:

    for col in pathways:
        scores[col] = (scores[col] - scores[col].mean()) / scores[col].std()

    return scores


# Label metabolic phenotype
def label_samples(scores: pd.DataFrame):

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


# Pie chart
def plot_label_distribution(scores: pd.DataFrame):

    label_counts = scores["label"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index,
            autopct='%1.1f%%', startangle=90)

    plt.title("Sample Label Distribution (ssGSEA)")
    plt.axis("equal")

    plt.savefig("label_distribution_pie.png", dpi=150)


# Add gene expression
def add_gene_expression(scores: pd.DataFrame, combined: pd.DataFrame):

    for gene in ["HK2", "UQCRC1", "RUNX1"]:
        scores[gene] = combined[(gene, "tpm_unstranded")]


# Scatter plots
def plot_scatter(scores: pd.DataFrame):

    fig, axes = plt.subplots(1, 9, figsize=(25, 5))

    pairs = [
        ("HALLMARK_GLYCOLYSIS", "HK2"),
        ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HK2"),
        ("delta", "HK2"),
        ("HALLMARK_GLYCOLYSIS", "UQCRC1"),
        ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "UQCRC1"),
        ("delta", "UQCRC1"),
        ("HALLMARK_GLYCOLYSIS", "RUNX1"),
        ("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "RUNX1"),
        ("delta", "RUNX1"),
    ]

    titles = [
        "Glycolysis vs HK2",
        "OXPHOS vs HK2",
        "Delta vs HK2",
        "Glycolysis vs UQCRC1",
        "OXPHOS vs UQCRC1",
        "Delta vs UQCRC1",
        "Glycolysis vs RUNX1",
        "OXPHOS vs RUNX1",
        "Delta vs RUNX1",
    ]

    for ax, (x, y), title in zip(axes, pairs, titles):

        ax.scatter(scores[x], scores[y], alpha=0.6)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("hk2_uqcrc1_scatter.png", dpi=150)

    def spearman_corr(x, y):
        return scores[[x, y]].corr(method="spearman").iloc[0, 1]

    print("Spearman Correlations:")

    for x, y in pairs:
        print(f"{x} vs {y}: {spearman_corr(x, y):.3f}")


# Delta distribution
def plot_delta_distribution(scores: pd.DataFrame):

    plt.figure(figsize=(7, 5))

    sns.histplot(scores["delta"], bins=20, kde=True)

    plt.xlabel("Delta (Glycolysis - OXPHOS)")
    plt.ylabel("Count")

    plt.title("Delta Distribution with Density Curve")

    plt.grid(True)

    plt.savefig("delta_histogram_kde.png", dpi=150)

    # plt.show()


# Prepare ML dataset
def prepare_ml_dataset(expr_matrix: pd.DataFrame, scores: pd.DataFrame):

    X = expr_matrix.T

    y = (scores["label"] == "Glycolytic").astype(int)

    df = X.copy()
    df["label"] = y

    return df


# Normalize molecular data
def normalize_molecular_data(df: pd.DataFrame):

    X = df.drop(columns=["label"])

    X = np.log1p(X)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    X_scaled["label"] = df["label"]

    return X_scaled


# Handle missing values
def handle_missing_data(df: pd.DataFrame):

    X = df.drop(columns=["label"])

    imputer = KNNImputer(n_neighbors=5)

    X_imputed = imputer.fit_transform(X)

    X_imputed = pd.DataFrame(
        X_imputed,
        columns=X.columns,
        index=X.index
    )

    X_imputed["label"] = df["label"]

    return X_imputed


# Remove low variance genes
def remove_low_variance(df: pd.DataFrame):

    X = df.drop(columns=["label"])

    selector = VarianceThreshold(threshold=0.01)

    X_reduced = selector.fit_transform(X)

    kept_genes = X.columns[selector.get_support()]

    X_reduced = pd.DataFrame(
        X_reduced,
        columns=kept_genes,
        index=X.index
    )

    X_reduced["label"] = df["label"]

    return X_reduced


# Train classifier
def train_model(df: pd.DataFrame):

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=12,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance")

    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    print(classification_report(y_test, preds))


    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(ascending=False)

    print("\nTop 20 Important Genes:")
    print(importance.head(20))

    print("\nHK2 rank:")
    print(importance.index.get_loc("HK2") + 1)


    # Plot feature importance
    top = importance.head(20)

    plt.figure(figsize=(8,6))
    sns.barplot(x=top.values, y=top.index)

    plt.title("Top 20 Important Genes (XGBoost)")
    plt.xlabel("Feature Importance")
    plt.ylabel("Gene")

    plt.tight_layout()

    plt.savefig("xgboost_feature_importance.png", dpi=150)

    return model

    
# Save outputs
def save_outputs(scores: pd.DataFrame, combined: pd.DataFrame):

    scores.to_csv("ssGSEA_scores_labeled.csv")

    subset = scores[["delta", "label"]].copy()

    subset.columns = pd.MultiIndex.from_product([["ssGSEA"], subset.columns])

    final = subset.join(combined)

    final.to_csv("tcga_kirc_combined_labeled.csv")


COMBINED_CACHE = Path("cache_combined.parquet")
EXPR_CACHE = Path("cache_expr_matrix.parquet")
# Main
def main():
    if COMBINED_CACHE.exists() and EXPR_CACHE.exists():
        print("Loading from cache...")
        combined = pd.read_parquet(COMBINED_CACHE)
        expr_matrix = pd.read_parquet(EXPR_CACHE)
    else:
        combined = load_all_samples(DATA_DIR)
        expr_matrix = build_expression_matrix(combined)
        combined.to_parquet(COMBINED_CACHE)
        expr_matrix.to_parquet(EXPR_CACHE)
        expr_matrix.to_csv("tpm_matrix.csv")
        
    print("Combined shape:", combined.shape)
    scores = run_ssgsea(expr_matrix, GMT_PATH)

    scores = normalize_scores(
        scores,
        ["HALLMARK_GLYCOLYSIS", "HALLMARK_OXIDATIVE_PHOSPHORYLATION"]
    )

    scores = label_samples(scores)

    print(scores["label"].value_counts())

    # Plot score distributions 
    plot_label_distribution(scores)

    add_gene_expression(scores, combined)
    plot_scatter(scores)

    plot_delta_distribution(scores)

    # Machine Learning Pipeline
    df = prepare_ml_dataset(expr_matrix, scores)

    df = normalize_molecular_data(df)

    df = handle_missing_data(df)

    df = remove_low_variance(df)

    train_model(df)

    save_outputs(scores, combined)


if __name__ == "__main__":
    main()