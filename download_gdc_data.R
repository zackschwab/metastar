install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")
library(TCGAbiolinks)
 
query <- GDCquery(
  project = "TCGA-KIRC",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
 
GDCdownload(query)
data <- GDCprepare(query)