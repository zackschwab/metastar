# MetaStar

METAbolic STAte Recognition (MetaStar) is a machine learning pipeline for predicting metabolic tumor states from TCGA-KIRC RNA-seq expression data using ssGSEA-derived pathway scores and XGBoost classification.

## Features

- Downloads TCGA-KIRC RNA-seq TPM expression data from the GDC portal
- Computes ssGSEA enrichment scores for glycolysis and oxidative phosphorylation (OXPHOS)
- Generates metabolic phenotype labels from pathway activity
- Performs correlation and biological validation analyses
- Trains and evaluates an XGBoost classifier
- Produces evaluation metrics and publication figures
- Caches processed datasets for faster reruns

## Requirements

- Python 3.10+
- R 4.0+

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Download Dataset

Download TCGA-KIRC RNA-seq data from the GDC portal:

```bash
Rscript download_gdc_data.R
```

This will create the `GDCdata/` directory containing the RNA-seq TPM expression files required for the pipeline.

## Run Pipeline

Execute the main pipeline:

```bash
python3 ssGSEA.py
```

The pipeline will:

1. Load and preprocess RNA-seq expression data
2. Compute ssGSEA pathway enrichment scores
3. Normalize pathway activity scores
4. Generate metabolic phenotype labels
5. Perform biological validation and correlation analysis
6. Train and evaluate the XGBoost classifier
7. Generate figures and save outputs

## Classification Setup

Samples are labeled using the delta between glycolysis and OXPHOS enrichment scores:

- `Glycolytic`: delta > 0.5
- `Oxidative`: delta < -0.5
- `Mixed`: −0.5 ≤ delta ≤ 0.5

Samples labeled as Mixed are excluded from binary classifier training.

## Outputs

Generated outputs are saved to the `results/` and `figures/` directories.

### results/

- `ssGSEA_scores_labeled.csv`: Sample IDs with glycolysis/OXPHOS NES scores, delta, and labels
- `tcga_kirc_combined_labeled.csv`: Merged ssGSEA labels and TPM expression
- `tpm_matrix.csv`: Gene-by-sample TPM matrix
- Cached expression matrices stored as parquet files for faster reruns

### figures/

- `label_distribution_pie_chart.png`: Metabolic phenotype distribution
- `delta_distribution_histogram.png`: Delta score distribution
- `gene_correlation_scatterplots.png`: 9-panel correlation grid (HK2, UQCRC1, RUNX1 vs. pathways)
- `confusion_matrix.png`: Classification performance on held-out test set

## Biological Validation

The pipeline performs additional biological validation analyses, including:

- HK2 statistical validation between metabolic states
- Spearman Correlation analysis of HK2, UQCRC1, and RUNX1 with pathway activity

## Dataset

This project uses RNA-seq data from:

- The Cancer Genome Atlas Kidney Renal Clear Cell Carcinoma (TCGA-KIRC)

## Model

The classifier is implemented using XGBoost with randomized hyperparameter search and evaluation on a held-out test set using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- Confusion matrix

## Citation

If you use MetaStar in your research, please cite our paper:

> **MetaStar: Predict tumor metabolic state from transcriptomic data**  
> Elias Mengistu, Zack Schwab  
