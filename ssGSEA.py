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
from sklearn.model_selection import StratifiedKFold

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
        # .fillna(0)
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


# Split data into train and test
def split_data(df: pd.DataFrame):

    X = df.drop(columns=["label"])
    y = df["label"]

    training_features, testing_features, training_answers, testing_answers  = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return training_features, testing_features, training_answers, testing_answers 


# Normalize molecular data
def normalize_molecular_data(training_features: pd.DataFrame, testing_features: pd.DataFrame):

    training_features = np.log1p(training_features)
    testing_features = np.log1p(testing_features)

    scaler = StandardScaler()

    training_features_scaled = scaler.fit_transform(training_features)
    testing_features_scaled = scaler.transform(testing_features)

    training_features_scaled = pd.DataFrame(
        training_features_scaled,
        columns=training_features.columns,
        index=training_features.index
    )

    testing_features_scaled = pd.DataFrame(
        testing_features_scaled,
        columns=testing_features.columns,
        index=testing_features.index
    )

    return training_features_scaled, testing_features_scaled


# Handle missing values
def handle_missing_data(training_features: pd.DataFrame, testing_features: pd.DataFrame):

    imputer = KNNImputer(n_neighbors=5)

    training_features_imputed = imputer.fit_transform(training_features)
    testing_features_imputed = imputer.transform(testing_features)

    training_features_imputed = pd.DataFrame(
        training_features_imputed,
        columns=training_features.columns,
        index=training_features.index
    )

    testing_features_imputed = pd.DataFrame(
        testing_features_imputed,
        columns=testing_features.columns,
        index=testing_features.index
    )

    return training_features_imputed, testing_features_imputed


# Remove low variance genes
def remove_low_variance(training_features: pd.DataFrame, testing_features: pd.DataFrame):

    selector = VarianceThreshold(threshold=0.01)

    training_features_reduced = selector.fit_transform(training_features)
    testing_features_reduced = selector.transform(testing_features)

    kept_genes = training_features.columns[selector.get_support()]

    training_features_reduced = pd.DataFrame(
        training_features_reduced,
        columns=kept_genes,
        index=training_features.index
    )

    testing_features_reduced = pd.DataFrame(
        testing_features_reduced,
        columns=kept_genes,
        index=testing_features.index
    )

    return training_features_reduced, testing_features_reduced


def cross_validate_model(df: pd.DataFrame, n_splits=5):

    X = df.drop(columns=["label"])
    y = df["label"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs = []
    accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        print(f"\nFold {fold + 1}")

        train_features, test_features = X.iloc[train_idx], X.iloc[val_idx]
        training_answers, testing_answers = y.iloc[train_idx], y.iloc[val_idx]

        # Preprocessing
        train_features = np.log1p(train_features)
        test_features = np.log1p(test_features)

        imputer = KNNImputer(n_neighbors=5)
        train_features = pd.DataFrame(
            imputer.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )
        test_features = pd.DataFrame(
            imputer.transform(test_features),
            columns=test_features.columns,
            index=test_features.index
        )

        selector = VarianceThreshold(threshold=0.01)
        train_features = pd.DataFrame(
            selector.fit_transform(train_features),
            columns=train_features.columns[selector.get_support()],
            index=train_features.index
        )
        test_features = pd.DataFrame(
            selector.transform(test_features),
            columns=train_features.columns,
            index=test_features.index
        )

        scaler = StandardScaler()
        train_features = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=train_features.columns,
            index=train_features.index
        )
        test_features = pd.DataFrame(
            scaler.transform(test_features),
            columns=train_features.columns,
            index=test_features.index
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

        model.fit(train_features, training_answers)

        preds = model.predict(test_features)
        probs = model.predict_proba(test_features)[:, 1]

        acc = accuracy_score(testing_answers, preds)
        auc = roc_auc_score(testing_answers, probs)

        print("Accuracy:", acc)
        print("AUC:", auc)

        accs.append(acc)
        aucs.append(auc)

    print("\n===== CV SUMMARY =====")
    print("Mean Accuracy:", np.mean(accs))
    print("Mean ROC-AUC:", np.mean(aucs))
    print("STD ROC-AUC:", np.std(aucs))

    
# Train classifier
def train_model(training_features: pd.DataFrame, testing_features: pd.DataFrame, training_answers: pd.Series, testing_answers : pd.Series):

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

    model.fit(training_features, training_answers)

    preds = model.predict(testing_features)
    probs = model.predict_proba(testing_features)[:, 1]

    print("\nModel Performance")

    print("Accuracy:", accuracy_score(testing_answers , preds))
    print("ROC-AUC:", roc_auc_score(testing_answers , probs))

    print(classification_report(testing_answers , preds))


    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=training_features.columns
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

    # Split data before normalizing to prevent data leakage
    # training_features, testing_features, training_answers, testing_answers  = split_data(df)
    # training_features, testing_features = normalize_molecular_data(training_features, testing_features)
    # training_features, testing_features = handle_missing_data(training_features, testing_features)
    # training_features, testing_features = remove_low_variance(training_features, testing_features)

    cross_validate_model(df)
    save_outputs(scores, combined)


if __name__ == "__main__":
    main()
