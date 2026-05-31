"""
Preprocessing module for drug response prediction.

This module replicates the exact preprocessing pipeline from the notebook:
1. Label encoding for categorical features
2. Feature selection to 9 positive features
3. Standard scaling

The module provides:
- Preprocessor class that can be fitted and transformed
- Save/load functionality for encoders and scaler
- Input validation
- Graceful handling of unseen categorical labels
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Any, Optional, Union
import warnings

logger = logging.getLogger(__name__)

# Constants from notebook analysis
CATEGORICAL_COLUMNS = [
    "CELL_LINE_NAME",
    "TCGA_DESC", 
    "DRUG_NAME",
    "TARGET",
    "TARGET_PATHWAY",
    "SITE",
    "HISTOLOGY",
    "GDSC_TISSUE_DESCRIPTOR_1",
    "GDSC_TISSUE_DESCRIPTOR_2",
    "CANCER_TYPE_MATCHING_TCGA_LABEL"
]

POSITIVE_FEATURES = [
    "AUC",
    "Z_SCORE", 
    "TARGET_ENC",
    "TARGET_PATHWAY_ENC",
    "TCGA_DESC_ENC",
    "GDSC_TISSUE_DESCRIPTOR_2_ENC",
    "CANCER_TYPE_MATCHING_TCGA_LABEL_ENC",
    "SITE_ENC",
    "GDSC_TISSUE_DESCRIPTOR_1_ENC"
]

NUMERIC_FEATURES = ["AUC", "Z_SCORE"]

# Unknown label sentinel value (used when a category wasn't seen during training)
UNKNOWN_LABEL_ENCODED = -1


class DrugResponsePreprocessor:
    """
    Preprocessor for drug response prediction data.
    
    This class replicates the exact preprocessing steps from the notebook:
    1. Label encoding for 10 categorical columns
    2. Feature selection to 9 positive features
    3. Standard scaling
    
    Handles unseen categorical labels gracefully by mapping them to a
    sentinel value (-1) instead of crashing.
    
    Attributes:
        label_encoders (Dict[str, LabelEncoder]): Dictionary of label encoders for each categorical column
        scaler (StandardScaler): Standard scaler for feature normalization
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(self):
        """Initialize the preprocessor with empty encoders and scaler."""
        self.label_encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'DrugResponsePreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: DataFrame with raw features including categorical and numeric columns
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info(f"Fitting preprocessor on data with shape {df.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Fill missing categorical values with "Unknown"
        for col in CATEGORICAL_COLUMNS:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).fillna("Unknown")
            else:
                raise ValueError(
                    f"Missing required categorical column: {col}. "
                    f"Available columns: {list(df_processed.columns)}"
                )
        
        # Fit label encoders
        for col, encoder in self.label_encoders.items():
            encoder.fit(df_processed[col])
            logger.debug(f"Fitted encoder for {col}: {len(encoder.classes_)} classes")
        
        # Transform to get encoded features
        df_encoded = self._encode_categoricals(df_processed)
        
        # Ensure all positive features are present
        missing_features = [f for f in POSITIVE_FEATURES if f not in df_encoded.columns]
        if missing_features:
            raise ValueError(f"Missing features after encoding: {missing_features}")
            
        # Fit scaler on the 9 positive features
        self.scaler.fit(df_encoded[POSITIVE_FEATURES])
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw data into preprocessed features.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            np.ndarray: Preprocessed features (n_samples, 9)
        """
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before transformation. "
                "Either load a saved preprocessor from disk or call fit() first. "
                "DO NOT fit a new preprocessor at API runtime."
            )
            
        logger.debug(
            f"[PREPROCESS] Transform input: shape={df.shape}, "
            f"columns={list(df.columns)}, sample_count={len(df)}"
        )
            
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Fill missing categorical values with "Unknown"
        missing_cols = []
        for col in CATEGORICAL_COLUMNS:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).fillna("Unknown")
            else:
                missing_cols.append(col)
        
        if missing_cols:
            logger.error(
                f"[PREPROCESS] Missing required categorical columns: {missing_cols}. "
                f"Available columns: {list(df_processed.columns)}"
            )
            raise ValueError(
                f"Missing required categorical columns: {missing_cols}. "
                f"Available columns: {list(df_processed.columns)}"
            )
        
        # Encode categorical features (handles unseen labels gracefully)
        df_encoded = self._encode_categoricals(df_processed)
        
        # Ensure all positive features are present
        missing_features = [f for f in POSITIVE_FEATURES if f not in df_encoded.columns]
        if missing_features:
            logger.error(f"[PREPROCESS] Missing features after encoding: {missing_features}")
            raise ValueError(f"Missing features after encoding: {missing_features}")
            
        # Scale features
        features = df_encoded[POSITIVE_FEATURES]
        logger.debug(
            f"[PREPROCESS] Before scaling: features_shape={features.shape}, "
            f"feature_names={POSITIVE_FEATURES}"
        )
        scaled_features = self.scaler.transform(features)
        
        logger.debug(
            f"[PREPROCESS] Transform output: shape={scaled_features.shape}, "
            f"dtype={scaled_features.dtype}, "
            f"min={scaled_features.min():.4f}, max={scaled_features.max():.4f}, "
            f"mean={scaled_features.mean():.4f}"
        )
        return scaled_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            np.ndarray: Preprocessed features (n_samples, 9)
        """
        return self.fit(df).transform(df)
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns using fitted label encoders.
        
        Handles unseen labels gracefully by mapping them to UNKNOWN_LABEL_ENCODED (-1)
        instead of raising ValueError. This is critical for production use where
        new cell lines, drugs, or targets may be encountered.
        
        Args:
            df: DataFrame with categorical columns filled as strings
            
        Returns:
            pd.DataFrame: DataFrame with encoded columns added
        """
        df_encoded = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                known_classes = set(encoder.classes_)
                encoded_values = []
                unseen_count = 0
                
                for val in df_encoded[col]:
                    if val in known_classes:
                        encoded_values.append(encoder.transform([val])[0])
                    else:
                        encoded_values.append(UNKNOWN_LABEL_ENCODED)
                        unseen_count += 1
                
                df_encoded[f"{col}_ENC"] = np.array(encoded_values, dtype=np.int64)
                
                if unseen_count > 0:
                    logger.warning(
                        f"Column '{col}': {unseen_count}/{len(df_encoded)} unseen labels "
                        f"mapped to {UNKNOWN_LABEL_ENCODED}. "
                        f"Example unseen: {[v for v in df_encoded[col].unique() if v not in known_classes][:3]}"
                    )
            else:
                raise ValueError(
                    f"Column {col} not found for encoding. "
                    f"Available columns: {list(df_encoded.columns)}"
                )
                
        return df_encoded
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the output features.
        
        Returns:
            List[str]: Names of the 9 positive features
        """
        return POSITIVE_FEATURES.copy()
    
    def get_known_classes(self, column: str) -> List[str]:
        """
        Get the known classes for a categorical column.
        
        Args:
            column: Column name
            
        Returns:
            List[str]: List of known class labels
        """
        if column in self.label_encoders:
            return list(self.label_encoders[column].classes_)
        return []
    
    def save(self, path: str) -> None:
        """
        Save the preprocessor to disk.
        
        Args:
            path: Directory path to save preprocessor components
        """
        os.makedirs(path, exist_ok=True)
        
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, os.path.join(path, f"{col}_encoder.pkl"))
            
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        
        # Save metadata
        metadata = {
            'is_fitted': self.is_fitted,
            'categorical_columns': CATEGORICAL_COLUMNS,
            'positive_features': POSITIVE_FEATURES,
            'unknown_label_encoded': UNKNOWN_LABEL_ENCODED
        }
        joblib.dump(metadata, os.path.join(path, "preprocessor_metadata.pkl"))
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DrugResponsePreprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            path: Directory path containing saved preprocessor components
            
        Returns:
            DrugResponsePreprocessor: Loaded preprocessor instance
        """
        logger.info(f"Loading preprocessor from {path}")
        preprocessor = cls()
        
        # Load label encoders
        for col in CATEGORICAL_COLUMNS:
            encoder_path = os.path.join(path, f"{col}_encoder.pkl")
            if os.path.exists(encoder_path):
                preprocessor.label_encoders[col] = joblib.load(encoder_path)
            else:
                raise FileNotFoundError(f"Encoder not found: {encoder_path}")
                
        # Load scaler
        scaler_path = os.path.join(path, "scaler.pkl")
        if os.path.exists(scaler_path):
            preprocessor.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
        # Load metadata
        metadata_path = os.path.join(path, "preprocessor_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            preprocessor.is_fitted = metadata.get('is_fitted', False)
        else:
            warnings.warn("Metadata not found, assuming preprocessor is fitted")
            preprocessor.is_fitted = True
            
        logger.info("Preprocessor loaded successfully")
        return preprocessor


def load_and_merge_training_data(data_dir: str = "data", max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge all data sources to create the full training dataset.
    
    This replicates the notebook's data merge process:
    1. Load GDSC_DATASET.csv (main drug response data)
    2. Load Cell_Lines_Details.xlsx → COSMIC tissue classification sheet (SITE, HISTOLOGY)
    3. Merge on COSMIC_ID to add SITE and HISTOLOGY columns
    4. Clean column names to uppercase with underscores
    
    Args:
        data_dir: Directory containing data files
        max_rows: Maximum number of rows to load (None = all)
        
    Returns:
        pd.DataFrame: Merged dataset with all required columns
    """
    logger.info(f"Loading training data from {data_dir}")
    
    # --- Step 1: Load main GDSC dataset ---
    gdsc_path = os.path.join(data_dir, "GDSC_DATASET.csv")
    if not os.path.exists(gdsc_path):
        raise FileNotFoundError(f"GDSC dataset not found: {gdsc_path}")
    
    nrows = max_rows if max_rows else None
    df = pd.read_csv(gdsc_path, nrows=nrows)
    logger.info(f"Loaded GDSC_DATASET.csv: {df.shape}")
    
    # --- Step 2: Load COSMIC tissue classification for SITE and HISTOLOGY ---
    excel_path = os.path.join(data_dir, "Cell_Lines_Details.xlsx")
    
    site_histology_available = False
    if os.path.exists(excel_path):
        try:
            cosmic_tissue = pd.read_excel(excel_path, sheet_name="COSMIC tissue classification")
            logger.info(f"Loaded COSMIC tissue classification: {cosmic_tissue.shape}")
            
            # Clean column names
            cosmic_tissue.columns = (
                cosmic_tissue.columns.astype(str)
                .str.strip()
                .str.upper()
                .str.replace("\n", "_")
                .str.replace(" ", "_")
            )
            
            # Rename to match expected format
            if "COSMIC_ID" in cosmic_tissue.columns:
                # Keep only needed columns
                merge_cols = ["COSMIC_ID"]
                if "SITE" in cosmic_tissue.columns:
                    merge_cols.append("SITE")
                if "HISTOLOGY" in cosmic_tissue.columns:
                    merge_cols.append("HISTOLOGY")
                
                if len(merge_cols) > 1:
                    cosmic_subset = cosmic_tissue[merge_cols].drop_duplicates(subset=["COSMIC_ID"])
                    df = pd.merge(df, cosmic_subset, on="COSMIC_ID", how="left")
                    site_histology_available = True
                    logger.info(f"Merged SITE/HISTOLOGY from COSMIC tissue classification")
        except Exception as e:
            logger.warning(f"Failed to load COSMIC tissue classification: {e}")
    else:
        logger.warning(f"Cell_Lines_Details.xlsx not found at {excel_path}")
    
    # --- Step 3: Clean all column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\n", "_")
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    
    # --- Step 4: Handle missing SITE/HISTOLOGY ---
    if "SITE" not in df.columns or "HISTOLOGY" not in df.columns:
        logger.warning(
            "SITE and/or HISTOLOGY columns not available from merge. "
            "Falling back to GDSC_TISSUE_DESCRIPTOR_1 → SITE, "
            "GDSC_TISSUE_DESCRIPTOR_2 → HISTOLOGY mapping."
        )
        if "SITE" not in df.columns and "GDSC_TISSUE_DESCRIPTOR_1" in df.columns:
            df["SITE"] = df["GDSC_TISSUE_DESCRIPTOR_1"].fillna("unknown")
            logger.info("Mapped GDSC_TISSUE_DESCRIPTOR_1 → SITE")
        if "HISTOLOGY" not in df.columns and "GDSC_TISSUE_DESCRIPTOR_2" in df.columns:
            df["HISTOLOGY"] = df["GDSC_TISSUE_DESCRIPTOR_2"].fillna("unknown")
            logger.info("Mapped GDSC_TISSUE_DESCRIPTOR_2 → HISTOLOGY")
    
    # Fill any remaining NaN in SITE/HISTOLOGY
    for col in ["SITE", "HISTOLOGY"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")
    
    # --- Step 5: Validate required columns ---
    required = CATEGORICAL_COLUMNS + NUMERIC_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"After data merge, still missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    
    logger.info(f"Final training data shape: {df.shape}")
    return df


def create_sample_dataframe() -> pd.DataFrame:
    """
    Create a sample DataFrame with the correct column structure.
    
    Returns:
        pd.DataFrame: Sample DataFrame with all required columns
    """
    # Sample data matching the structure of the GDSC dataset
    sample_data = {
        "CELL_LINE_NAME": ["A172", "A172", "A172"],
        "TCGA_DESC": ["GBM", "GBM", "GBM"],
        "DRUG_NAME": ["Camptothecin", "Erlotinib", "Rapamycin"],
        "TARGET": ["TOP1", "EGFR", "MTORC1"],
        "TARGET_PATHWAY": ["DNA replication", "EGFR signaling", "PI3K/MTOR signaling"],
        "SITE": ["central_nervous_system", "central_nervous_system", "central_nervous_system"],
        "HISTOLOGY": ["glioma", "glioma", "glioma"],
        "GDSC_TISSUE_DESCRIPTOR_1": ["nervous_system", "nervous_system", "nervous_system"],
        "GDSC_TISSUE_DESCRIPTOR_2": ["glioma", "glioma", "glioma"],
        "CANCER_TYPE_MATCHING_TCGA_LABEL": ["GBM", "GBM", "GBM"],
        "AUC": [0.93, 0.87, 0.91],
        "Z_SCORE": [0.43, -0.12, 0.21],
        "LN_IC50": [-1.46, -0.89, -1.12]  # Target variable
    }
    
    return pd.DataFrame(sample_data)


def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate that input DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    required_columns = CATEGORICAL_COLUMNS + NUMERIC_FEATURES
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
        
    return True


if __name__ == "__main__":
    # Example usage
    print("Testing DrugResponsePreprocessor...")
    
    # Create sample data
    df_sample = create_sample_dataframe()
    print(f"Sample data shape: {df_sample.shape}")
    print(f"Sample columns: {list(df_sample.columns)}")
    
    # Initialize preprocessor
    preprocessor = DrugResponsePreprocessor()
    
    # Fit and transform
    try:
        features = preprocessor.fit_transform(df_sample)
        print(f"Preprocessed features shape: {features.shape}")
        print(f"Feature names: {preprocessor.get_feature_names()}")
        
        # Test saving and loading
        preprocessor.save("test_preprocessor")
        print("Preprocessor saved successfully")
        
        loaded_preprocessor = DrugResponsePreprocessor.load("test_preprocessor")
        print("Preprocessor loaded successfully")
        
        # Test transformation with loaded preprocessor
        features_loaded = loaded_preprocessor.transform(df_sample)
        print(f"Loaded preprocessor features shape: {features_loaded.shape}")
        
        # Test unseen label handling
        df_unseen = df_sample.copy()
        df_unseen["CELL_LINE_NAME"] = "UNKNOWN_CELL_LINE_XYZ"
        df_unseen["DRUG_NAME"] = "UNKNOWN_DRUG_ABC"
        features_unseen = loaded_preprocessor.transform(df_unseen)
        print(f"Unseen label handling: features shape = {features_unseen.shape}")
        print(f"  CELL_LINE_NAME_ENC = {features_unseen[0][0]} (should include -1)")
        
        # Clean up test directory
        import shutil
        shutil.rmtree("test_preprocessor", ignore_errors=True)
        print("Test directory cleaned up")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
