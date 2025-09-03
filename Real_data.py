#This code show the diffferent genes regardless than the test statistics

import pandas as pd

# --- Configuration ---
# Define file paths and parameters here for easy modification
EXPRESSION_FILE = "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
CLINICAL_FILE = "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"
OUTPUT_FILE = 'tcga_brca_top_genes_4_subtypes.csv'

# Subtypes to keep for the final analysis
TARGET_SUBTYPES = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Her2', 'BRCA_Basal']
NUM_TOP_GENES = 5000

# --- 1. Load and Prepare Gene Expression Data ---
print("Loading gene expression data...")
# Load the gene expression file, setting the gene symbols as the index
expr = pd.read_csv(EXPRESSION_FILE, sep="\t", index_col=0)

# Transpose the DataFrame so that samples are rows and genes are columns
# Drop the 'Entrez_Gene_Id' column as it's not needed for the merge
expr_t = expr.drop(columns=["Entrez_Gene_Id"]).transpose()
print(f"Loaded expression data for {expr_t.shape[0]} samples and {expr_t.shape[1]} genes.")

# --- 2. Load Clinical Data ---
print("Loading clinical data...")
# Load the clinical data. Using low_memory=False to prevent DtypeWarning.
clinical = pd.read_csv(CLINICAL_FILE, sep="\t", low_memory=False)

# Keep a list of the original clinical columns to identify gene columns later
original_clinical_cols = clinical.columns.tolist()
print(f"Loaded clinical data for {clinical.shape[0]} patients.")

# --- 3. Merge Clinical and Expression Data ---
print("Merging clinical and expression datasets...")
# Merge the two dataframes using the sample ID.
# 'Sample ID' is in the clinical file, and the sample IDs are the index of the transposed expression file.
merged_df = pd.merge(clinical, expr_t, left_on="Sample ID", right_index=True, how="inner")
print(f"Merged dataset has {merged_df.shape[0]} samples after merging.")
# -------------------- THE FIX IS HERE --------------------
# First STEP: Remove any columns that are completely empty.
# This will find and delete the "Unnamed" phantom columns.
print("Removing completely empty columns from the dataset...")
merged_df.dropna(axis=1, how='all', inplace=True)
print("Empty columns removed.")
# -------------------- END OF FIX --------------------

# --- 4. Filter by Cancer Subtype ---
print(f"Filtering for target subtypes: {TARGET_SUBTYPES}...")
# Standardize the 'Subtype' column to remove any leading/trailing whitespace
merged_df['Subtype'] = merged_df['Subtype'].str.strip()

# Keep only the rows that match our target subtypes
filtered_df = merged_df[merged_df['Subtype'].isin(TARGET_SUBTYPES)].copy()
print(f"Found {filtered_df.shape[0]} samples belonging to the target subtypes.")

# --- 5. Select the Most Variable Genes ---
print(f"Identifying the top {NUM_TOP_GENES} most variable genes...")
# Identify gene columns as any column not originally in the clinical dataframe
gene_cols = [col for col in filtered_df.columns if col not in original_clinical_cols]

# Calculate the variance for each gene across the filtered samples
variances = filtered_df[gene_cols].var().sort_values(ascending=False)

# Get the names of the top N most variable genes
top_genes = variances.index[:NUM_TOP_GENES].tolist()
print("Top genes selected.")

# --- 6. Create and Export Final DataFrame ---
# Identify the clinical columns to keep (all non-gene columns in the filtered set)
final_clinical_cols = [col for col in filtered_df.columns if col not in gene_cols]

# Create the final dataframe with clinical columns plus the top variable gene columns
final_df = filtered_df[final_clinical_cols + top_genes]

# Export the final, processed data to a new CSV file
print(f"Exporting final data to {OUTPUT_FILE}...")
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- Process Complete ---")
print(f"Final file created: {OUTPUT_FILE}")
print(f"Number of samples in final file: {final_df.shape[0]}")

print(f"Number of columns in final file: {final_df.shape[1]} ({len(final_clinical_cols)} clinical + {len(top_genes)} genes)")
