"""Generate drug_auc_reference.csv from GDSC_DATASET.csv + Cell_Lines_Details.xlsx."""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Load GDSC dataset
gdsc = pd.read_csv(os.path.join(DATA_DIR, 'GDSC_DATASET.csv'))
print(f'GDSC shape: {gdsc.shape}')

# Load COSMIC tissue classification for SITE
cosmic = pd.read_excel(
    os.path.join(DATA_DIR, 'Cell_Lines_Details.xlsx'),
    sheet_name='COSMIC tissue classification'
)
cosmic.columns = cosmic.columns.str.strip()
cosmic_subset = cosmic[['COSMIC_ID', 'Site', 'Histology']].copy()
cosmic_subset.columns = ['COSMIC_ID', 'SITE', 'HISTOLOGY']
cosmic_subset = cosmic_subset.drop_duplicates(subset=['COSMIC_ID'])

# Merge GDSC with COSMIC to get SITE
merged = pd.merge(gdsc, cosmic_subset, on='COSMIC_ID', how='left')

# Fill missing SITE with GDSC Tissue descriptor 1 (fallback)
merged['SITE'] = merged['SITE'].fillna(merged['GDSC Tissue descriptor 1'])

# Build reference DataFrame
ref = pd.DataFrame({
    'DRUG_NAME': merged['DRUG_NAME'],
    'TARGET': merged['TARGET'],
    'TARGET_PATHWAY': merged['TARGET_PATHWAY'],
    'TCGA_DESC': merged['TCGA_DESC'],
    'GDSC_TISSUE_DESCRIPTOR_2': merged['GDSC Tissue descriptor 2'],
    'CANCER_TYPE_MATCHING_TCGA_LABEL': merged['Cancer Type (matching TCGA label)'],
    'SITE': merged['SITE'],
    'GDSC_TISSUE_DESCRIPTOR_1': merged['GDSC Tissue descriptor 1'],
    'AUC': merged['AUC']
})

# Drop rows with NaN in key columns
ref = ref.dropna(subset=[
    'DRUG_NAME', 'TARGET', 'TARGET_PATHWAY', 'TCGA_DESC',
    'GDSC_TISSUE_DESCRIPTOR_2', 'CANCER_TYPE_MATCHING_TCGA_LABEL',
    'SITE', 'GDSC_TISSUE_DESCRIPTOR_1', 'AUC'
])

# Standardize string columns to lowercase for consistent matching
for col in ['DRUG_NAME', 'TARGET', 'TARGET_PATHWAY', 'TCGA_DESC',
            'GDSC_TISSUE_DESCRIPTOR_2', 'CANCER_TYPE_MATCHING_TCGA_LABEL',
            'SITE', 'GDSC_TISSUE_DESCRIPTOR_1']:
    ref[col] = ref[col].astype(str).str.strip().str.lower()

print(f'Reference shape: {ref.shape}')
print(f'Unique drugs: {ref["DRUG_NAME"].nunique()}')
print(f'Sample rows:')
print(ref.head(3).to_string())

# Save
output_path = os.path.join(DATA_DIR, 'drug_auc_reference.csv')
ref.to_csv(output_path, index=False)
print(f'Saved to {output_path}')
print(f'File exists: {os.path.exists(output_path)}')