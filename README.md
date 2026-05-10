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
- `Mixed`: intermediate metabolic state

Samples labeled as Mixed are excluded from binary classifier training.

## Outputs

Generated outputs are saved to the `results/` and `figures/` directories.

### results/

- `ssGSEA_scores_labeled.csv`
- `tcga_kirc_combined_labeled.csv`
- `tpm_matrix.csv`
- Cached parquet files for faster reruns

### figures/

- `label_distribution_pie_chart.png`
- `delta_distribution_histogram.png`
- `gene_correlation_scatterplots.png`
- `confusion_matrix.png`

## Biological Validation

The pipeline performs additional biological validation analyses, including:

- HK2, UQCRC1, and RUN1 expression comparison between metabolic states
- Spearman correlation analysis between pathway scores and selected marker genes

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
