# metastar
METAbolic STAte Recognition machine learning model based on multi-omic biological data. 

## Getting started

```bash
git clone https://github.com/zackschwab/metastar.git
cd metastar
```

### Download the GDC data by running this R code
```bash
Rscript download_gdc_data.R
```

Copy the GDC data into the metastar directory

```bash
pip install -r requirements.txt
python3 ./ssGSEA.py
```