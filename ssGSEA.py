import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Config
DATA_DIR = Path("./GDCdata/TCGA-KIRC/Transcriptome_Profiling/Gene_Expression_Quantification")
GMT_PATH = "./HALLMARK_COMBINED.gmt"
EXPR_COLS = ["tpm_unstranded"]

# Clean and load a single sample TSV file
def load_sample(fpath: Path) -> Optional[pd.Series]:

    df = pd.read_csv(fpath, sep="\t", comment="#")

    df = df[~df["gene_id"].str.startswith("N_")]
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

# Plot the distribution of sample labels
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

# Plots tpm_unstranded expression of selected genes
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

    # Nice Spearman table
    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│              Spearman Correlations                         |")
    print("├────────────────────────────────────────────────────────────┤")
    print("│  {:<45} │  {:>6}  │".format("Comparison", "ρ  "))
    print("├────────────────────────────────────────────────────────────┤")

    for (x, y) in pairs:
        rho = scores[[x, y]].corr(method="spearman").iloc[0, 1]
        label = f"{x} vs {y}"
        print("│  {:<45} │  {:>+6.3f}   │".format(label, rho))

    print("└────────────────────────────────────────────────────────────┘")

# Delta distribution
def plot_delta_distribution(scores: pd.DataFrame):

    plt.figure(figsize=(7, 5))

    sns.histplot(scores["delta"], bins=20, kde=True)

    plt.xlabel("Delta (Glycolysis - OXPHOS)")
    plt.ylabel("Count")

    plt.title("Delta Distribution with Density Curve")

    plt.grid(True)

    plt.savefig("delta_histogram_kde.png", dpi=150)

# Prepare the dataset for machine learning in a format suitable for training
def prepare_ml_dataset(expr_matrix, scores, binary=True):
    if binary:
        mask = scores["label"].isin(["Glycolytic", "Oxidative"])
        scores_subset = scores.loc[mask].copy()
        expr_subset = expr_matrix.loc[:, mask].copy()
    else:
        scores_subset = scores.copy()
        expr_subset = expr_matrix.copy()

    X = expr_subset.T
    y = (scores_subset["label"] == "Glycolytic").astype(int)
    df = X.copy()
    df["label"] = y

    print("\n┌─────────────────────────────────────┐")
    print("│         ML Dataset Summary          │")
    print("├─────────────────────────────────────┤")
    print(f"│  Total samples       :  {df.shape[0]:>4}        │")
    print(f"│  Features (genes)    :  {df.shape[1]-1:>4}       │")
    print("├─────────────────────────────────────┤")
    print(f"│  Glycolytic (1)      :  {y.sum():>4}  ({100*y.sum()/len(y):.1f}%)│")
    print(f"│  Oxidative  (0)      :  {(y==0).sum():>4}  ({100*(y==0).sum()/len(y):.1f}%)│")
    print("└─────────────────────────────────────┘")

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

# Hyperparameter tuning function
def tune_hyperparameters(training_features: pd.DataFrame, training_answers: pd.Series) -> Dict:
    param_dist = {
        'n_estimators': [100, 300, 500, 800, 1200],
        'max_depth': [2, 3, 4, 5, 6, 8],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10], 
        'min_child_weight': [1, 3, 5, 7, 10]
    }

    base_model = XGBClassifier(
        random_state=42,
        n_jobs=12,
        eval_metric="logloss"
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=1,
        cv=2,
        scoring='roc_auc',
        n_jobs=12,
        random_state=42,
        verbose=1,
        refit=False,
        return_train_score=True
    )

    print("\n>>> Running hyperparameter search...")
    search.fit(training_features, training_answers)

    results = search.cv_results_
    n_folds = search.cv if isinstance(search.cv, int) else search.cv.n_splits

    print("\n--- Iteration Results ---")
    for i in range(search.n_iter):
        mean_s = results["mean_test_score"][i]
        fold_scores = [results[f"split{j}_test_score"][i] for j in range(n_folds)]
        params = results["params"][i]
        marker = "  <-- BEST" if i == search.best_index_ else ""
        print(f"Iter {i+1}: mean={mean_s:.4f} | folds={fold_scores} | {params}{marker}")

    best_idx = search.best_index_
    test_scores = [results[f"split{j}_test_score"][best_idx] for j in range(n_folds)]
    train_scores = [results[f"split{j}_train_score"][best_idx] for j in range(n_folds)]
    best_fold = int(np.argmax(test_scores)) + 1
    gap = np.mean(train_scores) - search.best_score_

    print(f"\nBest combo: mean test ROC-AUC = {search.best_score_:.4f}")
    print(f"  Best fold : Fold {best_fold} ({max(test_scores):.4f})")
    print(f"  Train gap : {gap:.4f}  {'(possible overfit)' if gap > 0.05 else ''}")

    return search.best_params_

# Final model training with best hyperparameters
def train_model(training_features: pd.DataFrame, testing_features: pd.DataFrame,
                training_answers: pd.Series, testing_answers: pd.Series,
                best_params: Optional[Dict] = None):

    # If no tuned params provided, fall back to sensible defaults
    if best_params is None:
        best_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=12,
        eval_metric="logloss"
    )

    model.fit(training_features, training_answers)

    preds = model.predict(testing_features)
    probs = model.predict_proba(testing_features)[:, 1]

    print("\n=== Final Tuned Model Performance ===")
    print("Accuracy:", accuracy_score(testing_answers, preds))
    print("ROC-AUC:", roc_auc_score(testing_answers, probs))
    print(classification_report(testing_answers, preds))

    return model
  
# Save outputs
def save_outputs(scores: pd.DataFrame, combined: pd.DataFrame):

    scores.to_csv("ssGSEA_scores_labeled.csv")

    subset = scores[["delta", "label"]].copy()

    subset.columns = pd.MultiIndex.from_product([["ssGSEA"], subset.columns])

    final = subset.join(combined)

    final.to_csv("tcga_kirc_combined_labeled.csv")


# Check if HK2 is elevated in Glycolytic vs Oxidative samples
def validate_hk2_signal(scores: pd.DataFrame):
    glyco = scores.loc[scores["label"] == "Glycolytic", "HK2"]
    oxi  = scores.loc[scores["label"] == "Oxidative",  "HK2"]

    if len(glyco) == 0 or len(oxi) == 0:
        print("  Warning: Missing Glycolytic or Oxidative samples for HK2 validation.")
        return

    stat, p = mannwhitneyu(glyco, oxi, alternative='greater')

    print("\n┌─────────────────────────────────────┐")
    print("│     HK2 Biological Validation       │")
    print("├─────────────────────────────────────┤")
    print(f"│  Glycolytic median :  {glyco.median():>8.2f} TPM  │")
    print(f"│  Oxidative median  :  {oxi.median():>8.2f} TPM  │")
    print(f"│  Fold change       :  {glyco.median() / oxi.median():>8.1f}x      │")
    print("├─────────────────────────────────────┤")
    print(f"│  Mann-Whitney U p  :  {p:.2e}       │")
    print(f"│  Significant?      :  {'YES ✓' if p < 0.05 else 'NO ✗':>15}│")
    print("└─────────────────────────────────────┘\n")

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
        print("Loaded combined shape:", combined.shape)
        expr_matrix = build_expression_matrix(combined)
        combined.to_parquet(COMBINED_CACHE)
        expr_matrix.to_parquet(EXPR_CACHE)
        expr_matrix.to_csv("tpm_matrix.csv")

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
    validate_hk2_signal(scores)
    plot_delta_distribution(scores)

    # Machine Learning Pipeline
    df = prepare_ml_dataset(expr_matrix, scores, binary=True)

    # Split data for training and testing
    training_features, testing_features, training_answers, testing_answers = split_data(df)

    # Hyperparameter tuning on the preprocessed training set
    best_params = tune_hyperparameters(training_features, training_answers)

    # Final model: uses best params from tuning, trained on full train, evaluated on held-out test
    final_model = train_model(
        training_features,
        testing_features,
        training_answers,
        testing_answers,
        best_params=best_params
    )

    save_outputs(scores, combined)


if __name__ == "__main__":
    main()