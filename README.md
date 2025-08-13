# Code for: A Robust Geometric Permutation Test for Equality of Covariance Matrices

This repository contains the Python code for the real-data analysis in the manuscript "A Robust Geometric Permutation Test for Equality of Covariance Matrices in High-Dimensional Data" by Saleh Valenti.

## Data Source

The real-world application data used in this study is publicly available from The Cancer Genome Atlas (TCGA) project. The specific raw gene expression and clinical data files can be accessed via the cBioPortal for Cancer Genomics (https://www.cbioportal.org/). Our analysis was performed on the **Breast Invasive Carcinoma (TCGA, Pan-Cancer Atlas)** dataset.

## Requirements

The analysis was performed using Python 3. The following libraries are required and can be installed via `pip`:
- `numpy`
- `pandas`
- `scipy`
- `pingouin`
- `numba`
- `tqdm`
- `scikit-learn` (for the imputer, though not used in the final clean run)

## How to Reproduce the Results

To reproduce the analysis and the results in Table 1 of the manuscript, follow these steps in order:

### Step 1: Preprocessing

Run the data preprocessing script to download the raw data, filter it, and create the final analysis file.

```bash
python real_data.py ```
This will produce the file `tcga_brca_top_genes_4_subtypes.csv`.

*(Note: Please ensure the names `real_data.py` match your preprocessor script name)*

### Step 2: Main Analysis

Run the main analysis script on the file generated in the previous step.

```bash
python real_data_analysis.py
