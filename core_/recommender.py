"""
Drug recommendation engine.

This module provides functionality to:
1. Load drug database from Compounds-annotation.csv
2. Load AUC reference data from drug_auc_reference.csv
3. Generate drug recommendations for a given cell line using cancer context filtering
4. Rank drugs by predicted effectiveness with response categories
5. Compute IC50 scores and percentages
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import warnings
import os
import logging

from .predictor import DrugResponsePredictor
from .preprocessing import CATEGORICAL_COLUMNS

logger = logging.getLogger(__name__)

# Response category thresholds (based on LN_IC50)
CATEGORY_ORDER = {
    "Very Sensitive": 0,
    "Sensitive": 1,
    "Moderate": 2,
    "Resistant": 3,
}


def classify_response(ln_ic50: float) -> str:
    """
    Determine response category from Predicted_LN_IC50.

    Categories:
    - LN_IC50 < -2          → Very Sensitive
    - -2 ≤ LN_IC50 < 0      → Sensitive
    - 0 ≤ LN_IC50 < 2       → Moderate
    - LN_IC50 ≥ 2           → Resistant

    Args:
        ln_ic50: Predicted LN_IC50 value

    Returns:
        Response category string
    """
    if ln_ic50 < -2:
        return "Very Sensitive"
    elif ln_ic50 < 0:
        return "Sensitive"
    elif ln_ic50 < 2:
        return "Moderate"
    else:
        return "Resistant"


class DrugRecommender:
    """
    Drug recommendation engine for cancer cell lines.

    This class:
    - Loads drug information from Compounds-annotation.csv
    - Loads AUC reference data from drug_auc_reference.csv
    - For a given cell line cancer context, filters AUC reference to find matching drugs
    - Predicts LN_IC50 for each drug using the regression model
    - Computes Predicted_IC50 = exp(Predicted_LN_IC50)
    - Classifies response category from LN_IC50
    - Computes IC50_Score and IC50_Score_Percentage
    - Deduplicates by DRUG_NAME
    - Ranks drugs by category, score, and IC50

    Attributes:
        drug_df (pd.DataFrame): DataFrame of drug information from Compounds-annotation.csv
        auc_ref_df (pd.DataFrame): DataFrame of AUC reference data
        predictor (DrugResponsePredictor): Predictor instance
        default_auc (float): Default AUC value for drugs (fallback)
        default_z_score (float): Default Z_SCORE value for drugs
    """

    def __init__(self,
                 predictor: Optional[DrugResponsePredictor] = None,
                 drug_data_path: str = "data/Compounds-annotation.csv",
                 auc_ref_path: str = "data/drug_auc_reference.csv"):
        """
        Initialize the drug recommender.

        Args:
            predictor: DrugResponsePredictor instance (creates new one if None)
            drug_data_path: Path to drug annotation CSV file
            auc_ref_path: Path to AUC reference CSV file
        """
        self.predictor = predictor or DrugResponsePredictor()
        self.drug_data_path = drug_data_path
        self.auc_ref_path = auc_ref_path
        self.drug_df = None
        self.auc_ref_df = None
        self.default_auc = 0.5  # Neutral default value (fallback)
        self.default_z_score = 0.0  # Neutral default value

        # Load drug data and AUC reference
        self._load_drug_data()
        self._load_auc_reference()

    def _load_drug_data(self) -> None:
        """Load drug data from CSV file."""
        try:
            if os.path.exists(self.drug_data_path):
                self.drug_df = pd.read_csv(self.drug_data_path)
                logger.info(f"Loaded {len(self.drug_df)} drugs from {self.drug_data_path}")
            else:
                logger.warning(f"Drug data file not found: {self.drug_data_path}")
                self.drug_df = self._create_minimal_drug_df()
        except Exception as e:
            logger.warning(f"Failed to load drug data: {e}")
            self.drug_df = self._create_minimal_drug_df()

    def _load_auc_reference(self) -> None:
        """Load AUC reference data from CSV file."""
        try:
            if os.path.exists(self.auc_ref_path):
                self.auc_ref_df = pd.read_csv(self.auc_ref_path)
                # Ensure string columns are lowercase for consistent matching
                string_cols = [
                    'DRUG_NAME', 'TARGET', 'TARGET_PATHWAY', 'TCGA_DESC',
                    'GDSC_TISSUE_DESCRIPTOR_2', 'CANCER_TYPE_MATCHING_TCGA_LABEL',
                    'SITE', 'GDSC_TISSUE_DESCRIPTOR_1'
                ]
                for col in string_cols:
                    if col in self.auc_ref_df.columns:
                        self.auc_ref_df[col] = self.auc_ref_df[col].astype(str).str.strip().str.lower()
                logger.info(f"Loaded {len(self.auc_ref_df)} AUC reference rows from {self.auc_ref_path}")
            else:
                logger.warning(f"AUC reference file not found: {self.auc_ref_path}")
                self.auc_ref_df = None
        except Exception as e:
            logger.warning(f"Failed to load AUC reference: {e}")
            self.auc_ref_df = None

    def _create_minimal_drug_df(self) -> pd.DataFrame:
        """Create a minimal drug dataframe for testing."""
        drugs = [
            {"DRUG_ID": 1, "DRUG_NAME": "Erlotinib", "TARGET": "EGFR", "TARGET_PATHWAY": "EGFR signaling"},
            {"DRUG_ID": 3, "DRUG_NAME": "Rapamycin", "TARGET": "MTORC1", "TARGET_PATHWAY": "PI3K/MTOR signaling"},
            {"DRUG_ID": 5, "DRUG_NAME": "Sunitinib", "TARGET": "PDGFR", "TARGET_PATHWAY": "RTK signaling"},
            {"DRUG_ID": 11, "DRUG_NAME": "Paclitaxel", "TARGET": "Microtubule stabiliser", "TARGET_PATHWAY": "Mitosis"},
            {"DRUG_ID": 1003, "DRUG_NAME": "Camptothecin", "TARGET": "TOP1", "TARGET_PATHWAY": "DNA replication"},
        ]
        return pd.DataFrame(drugs)

    def _lookup_used_auc(self, drug_name: str, target: str, target_pathway: str,
                         tcga_desc: str, gdsc_tissue_desc_2: str,
                         cancer_type: str, site: str,
                         gdsc_tissue_desc_1: str) -> Optional[float]:
        """
        Look up Used_AUC from the AUC reference data.

        Filters rows matching all 8 context fields and returns the mean AUC.

        Args:
            drug_name: Drug name (will be lowercased)
            target: Drug target (will be lowercased)
            target_pathway: Target pathway (will be lowercased)
            tcga_desc: TCGA description (will be lowercased)
            gdsc_tissue_desc_2: GDSC tissue descriptor 2 (will be lowercased)
            cancer_type: Cancer type matching TCGA label (will be lowercased)
            site: Site (will be lowercased)
            gdsc_tissue_desc_1: GDSC tissue descriptor 1 (will be lowercased)

        Returns:
            Mean AUC from matching rows, or None if no matches found
        """
        if self.auc_ref_df is None:
            return None

        # Standardize input values to lowercase for matching
        match_values = {
            'DRUG_NAME': str(drug_name).strip().lower(),
            'TARGET': str(target).strip().lower(),
            'TARGET_PATHWAY': str(target_pathway).strip().lower(),
            'TCGA_DESC': str(tcga_desc).strip().lower(),
            'GDSC_TISSUE_DESCRIPTOR_2': str(gdsc_tissue_desc_2).strip().lower(),
            'CANCER_TYPE_MATCHING_TCGA_LABEL': str(cancer_type).strip().lower(),
            'SITE': str(site).strip().lower(),
            'GDSC_TISSUE_DESCRIPTOR_1': str(gdsc_tissue_desc_1).strip().lower(),
        }

        # Filter AUC reference by all 8 context fields
        mask = pd.Series([True] * len(self.auc_ref_df), index=self.auc_ref_df.index)
        for col, val in match_values.items():
            if col in self.auc_ref_df.columns:
                mask = mask & (self.auc_ref_df[col] == val)

        matched = self.auc_ref_df[mask]

        if len(matched) == 0:
            return None

        used_auc = matched['AUC'].mean()
        return float(used_auc)

    def recommend(self,
                  cell_line_data: Dict[str, Any],
                  max_drugs: Optional[int] = None,
                  model_name: Optional[str] = None,
                  include_score: bool = True) -> List[Dict[str, Any]]:
        """
        Generate drug recommendations for a cell line.

        Enhanced workflow:
        1. For each drug in the database, look up Used_AUC from AUC reference
           using cancer context fields
        2. Skip drugs with no matching AUC reference rows
        3. Predict LN_IC50 using the regression model with Used_AUC
        4. Compute Predicted_IC50 = exp(Predicted_LN_IC50)
        5. Classify response category from LN_IC50
        6. Compute IC50_Score = 1/(1+Predicted_IC50)
        7. Compute IC50_Score_Percentage = IC50_Score * 100
        8. Deduplicate by DRUG_NAME
        9. Rank by category order, then IC50_Score_Percentage desc,
           then Predicted_IC50 asc, then DRUG_NAME asc

        Args:
            cell_line_data: Dictionary with cell line features (must include all categorical columns)
            max_drugs: Maximum number of drugs to return (None = up to 100)
            model_name: Name of model to use for prediction (uses default if None)
            include_score: Whether to include recommendation score

        Returns:
            List of dictionaries with drug recommendations, sorted by ranking
        """
        # Validate cell line data
        self._validate_cell_line_data(cell_line_data)

        # Get available models
        available_models = self.predictor.get_available_models()
        if not available_models:
            logger.warning("No models available for recommendation")
            return []

        # Use specified model or default
        model_to_use = model_name or self.predictor.default_model

        # Extract cancer context fields from cell_line_data for AUC lookup
        tcga_desc = cell_line_data.get('TCGA_DESC', '')
        site = cell_line_data.get('SITE', '')
        gdsc_tissue_desc_1 = cell_line_data.get('GDSC_TISSUE_DESCRIPTOR_1', '')
        gdsc_tissue_desc_2 = cell_line_data.get('GDSC_TISSUE_DESCRIPTOR_2', '')
        cancer_type = cell_line_data.get('CANCER_TYPE_MATCHING_TCGA_LABEL', '')

        # Prepare recommendations list
        recommendations = []
        seen_drugs = set()  # For deduplication

        # For each drug, look up Used_AUC and predict
        for _, drug_row in self.drug_df.iterrows():
            drug_name = drug_row['DRUG_NAME']
            target = drug_row.get('TARGET', '')
            target_pathway = drug_row.get('TARGET_PATHWAY', '')

            # Deduplicate: skip if we've already processed this drug
            drug_name_lower = str(drug_name).strip().lower()
            if drug_name_lower in seen_drugs:
                continue
            seen_drugs.add(drug_name_lower)

            # Look up Used_AUC from reference data
            used_auc = self._lookup_used_auc(
                drug_name=drug_name,
                target=target,
                target_pathway=target_pathway,
                tcga_desc=tcga_desc,
                gdsc_tissue_desc_2=gdsc_tissue_desc_2,
                cancer_type=cancer_type,
                site=site,
                gdsc_tissue_desc_1=gdsc_tissue_desc_1
            )

            # If no matching AUC reference rows, skip this drug
            if used_auc is None:
                logger.debug(f"No AUC reference match for drug '{drug_name}' with given cancer context")
                continue

            # Create input data combining cell line and drug features, with Used_AUC
            input_data = self._create_input_data(cell_line_data, drug_row, used_auc=used_auc)

            try:
                # Make prediction — model output is LN_IC50
                result = self.predictor.predict(input_data, model_name=model_to_use)
                predicted_ln_ic50 = float(result['predictions'][0])

                # Compute Predicted_IC50 = exp(Predicted_LN_IC50)
                predicted_ic50 = float(np.exp(predicted_ln_ic50))

                # Determine response category
                response_category = classify_response(predicted_ln_ic50)

                # Compute IC50_Score = 1 / (1 + Predicted_IC50)
                ic50_score = float(1.0 / (1.0 + predicted_ic50))

                # Compute IC50_Score_Percentage = IC50_Score * 100
                ic50_score_percentage = float(ic50_score * 100)

                # Create recommendation entry
                rec = {
                    'drug_id': int(drug_row.get('DRUG_ID', 0)),
                    'drug_name': drug_name,
                    'target': target,
                    'target_pathway': target_pathway,
                    'predicted_ln_ic50': predicted_ln_ic50,
                    'predicted_ic50': predicted_ic50,
                    'response_category': response_category,
                    'ic50_score': ic50_score,
                    'ic50_score_percentage': ic50_score_percentage,
                    'used_auc': used_auc,
                    'model_used': model_to_use
                }

                # Add legacy score if requested (for backward compatibility)
                if include_score:
                    rec['score'] = ic50_score

                recommendations.append(rec)

            except Exception as e:
                logger.warning(f"Failed to predict for drug {drug_name}: {e}")
                continue

        if not recommendations:
            logger.warning("No recommendations generated — no drugs matched the cancer context in AUC reference")
            return []

        # Sort by ranking criteria:
        # 1. Category order (Very Sensitive > Sensitive > Moderate > Resistant)
        # 2. Higher IC50_Score_Percentage
        # 3. Lower Predicted_IC50
        # 4. DRUG_NAME ascending (tie breaker)
        recommendations.sort(key=lambda x: (
            CATEGORY_ORDER.get(x['response_category'], 99),
            -x['ic50_score_percentage'],
            x['predicted_ic50'],
            str(x['drug_name']).lower()
        ))

        # Add ranking
        for i, rec in enumerate(recommendations, 1):
            rec['rank'] = i

        # Limit to max_drugs if specified, default 100
        limit = max_drugs if max_drugs is not None else 100
        recommendations = recommendations[:limit]

        return recommendations

    def recommend_batch(self,
                        cell_line_data: Dict[str, Any],
                        drug_names: List[str],
                        model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate recommendations for specific drugs only.

        Args:
            cell_line_data: Dictionary with cell line features
            drug_names: List of drug names to evaluate
            model_name: Name of model to use for prediction

        Returns:
            List of dictionaries with drug recommendations
        """
        # Filter drug dataframe to requested drugs
        drug_subset = self.drug_df[self.drug_df['DRUG_NAME'].isin(drug_names)]

        if len(drug_subset) == 0:
            logger.warning(f"No matching drugs found for: {drug_names}")
            return []

        # Extract cancer context fields
        tcga_desc = cell_line_data.get('TCGA_DESC', '')
        site = cell_line_data.get('SITE', '')
        gdsc_tissue_desc_1 = cell_line_data.get('GDSC_TISSUE_DESCRIPTOR_1', '')
        gdsc_tissue_desc_2 = cell_line_data.get('GDSC_TISSUE_DESCRIPTOR_2', '')
        cancer_type = cell_line_data.get('CANCER_TYPE_MATCHING_TCGA_LABEL', '')

        recommendations = []
        seen_drugs = set()

        for _, drug_row in drug_subset.iterrows():
            drug_name = drug_row['DRUG_NAME']
            target = drug_row.get('TARGET', '')
            target_pathway = drug_row.get('TARGET_PATHWAY', '')

            # Deduplicate
            drug_name_lower = str(drug_name).strip().lower()
            if drug_name_lower in seen_drugs:
                continue
            seen_drugs.add(drug_name_lower)

            # Look up Used_AUC
            used_auc = self._lookup_used_auc(
                drug_name=drug_name,
                target=target,
                target_pathway=target_pathway,
                tcga_desc=tcga_desc,
                gdsc_tissue_desc_2=gdsc_tissue_desc_2,
                cancer_type=cancer_type,
                site=site,
                gdsc_tissue_desc_1=gdsc_tissue_desc_1
            )

            if used_auc is None:
                logger.debug(f"No AUC reference match for drug '{drug_name}' in batch request")
                continue

            input_data = self._create_input_data(cell_line_data, drug_row, used_auc=used_auc)

            try:
                result = self.predictor.predict(input_data, model_name=model_name)
                predicted_ln_ic50 = float(result['predictions'][0])
                predicted_ic50 = float(np.exp(predicted_ln_ic50))
                response_category = classify_response(predicted_ln_ic50)
                ic50_score = float(1.0 / (1.0 + predicted_ic50))
                ic50_score_percentage = float(ic50_score * 100)

                rec = {
                    'drug_name': drug_name,
                    'target': target,
                    'target_pathway': target_pathway,
                    'predicted_ln_ic50': predicted_ln_ic50,
                    'predicted_ic50': predicted_ic50,
                    'response_category': response_category,
                    'ic50_score': ic50_score,
                    'ic50_score_percentage': ic50_score_percentage,
                    'used_auc': used_auc,
                    'score': ic50_score,
                    'model_used': model_name or self.predictor.default_model
                }

                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Failed to predict for drug {drug_name}: {e}")
                continue

        # Sort by ranking criteria
        recommendations.sort(key=lambda x: (
            CATEGORY_ORDER.get(x['response_category'], 99),
            -x['ic50_score_percentage'],
            x['predicted_ic50'],
            str(x['drug_name']).lower()
        ))

        # Add ranking
        for i, rec in enumerate(recommendations, 1):
            rec['rank'] = i

        return recommendations

    def _create_input_data(self,
                          cell_line_data: Dict[str, Any],
                          drug_row: pd.Series,
                          used_auc: Optional[float] = None) -> Dict[str, Any]:
        """
        Create input data dictionary combining cell line and drug features.

        Args:
            cell_line_data: Cell line features
            drug_row: Drug information row
            used_auc: Used_AUC value from reference lookup (overrides default)

        Returns:
            Combined input data dictionary
        """
        input_data = cell_line_data.copy()

        # Add drug-specific features
        input_data['DRUG_NAME'] = drug_row['DRUG_NAME']
        input_data['TARGET'] = drug_row.get('TARGET', 'Unknown')
        input_data['TARGET_PATHWAY'] = drug_row.get('TARGET_PATHWAY', 'Unknown')

        # Use Used_AUC if provided, otherwise fall back to default
        if used_auc is not None:
            input_data['AUC'] = used_auc
        elif 'AUC' not in input_data:
            input_data['AUC'] = self.default_auc

        # Add default Z_SCORE if not present
        if 'Z_SCORE' not in input_data:
            input_data['Z_SCORE'] = self.default_z_score

        return input_data

    def _validate_cell_line_data(self, cell_line_data: Dict[str, Any]) -> None:
        """
        Validate that cell line data has all required categorical columns.

        Args:
            cell_line_data: Dictionary to validate

        Raises:
            ValueError: If required columns are missing
        """
        # Required cell line specific columns (excluding drug-specific ones)
        required_cell_line_cols = [
            'CELL_LINE_NAME', 'TCGA_DESC', 'SITE', 'HISTOLOGY',
            'GDSC_TISSUE_DESCRIPTOR_1', 'GDSC_TISSUE_DESCRIPTOR_2',
            'CANCER_TYPE_MATCHING_TCGA_LABEL'
        ]

        missing_cols = [col for col in required_cell_line_cols if col not in cell_line_data]
        if missing_cols:
            raise ValueError(f"Missing required cell line columns: {missing_cols}")

    def _calculate_score(self, ic50: float) -> float:
        """
        Calculate recommendation score from IC50.

        Legacy method for backward compatibility.
        Formula: score = 1 / (1 + abs(ic50))

        Args:
            ic50: Predicted IC50 value

        Returns:
            Recommendation score (higher = better)
        """
        denominator = 1 + abs(ic50)
        score = 1.0 / denominator
        return min(max(score, 0.0), 1.0)

    def get_drug_count(self) -> int:
        """
        Get number of drugs in the database.

        Returns:
            Number of drugs
        """
        return len(self.drug_df) if self.drug_df is not None else 0

    def get_drug_names(self) -> List[str]:
        """
        Get list of all drug names.

        Returns:
            List of drug names
        """
        if self.drug_df is not None and 'DRUG_NAME' in self.drug_df.columns:
            return self.drug_df['DRUG_NAME'].tolist()
        return []

    def get_drug_info(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific drug.

        Args:
            drug_name: Name of the drug

        Returns:
            Dictionary with drug information or None if not found
        """
        if self.drug_df is None:
            return None

        drug_row = self.drug_df[self.drug_df['DRUG_NAME'] == drug_name]
        if len(drug_row) == 0:
            return None

        return drug_row.iloc[0].to_dict()


def create_sample_recommendation() -> None:
    """Create and test a sample recommendation."""
    print("Testing DrugRecommender...")

    # Create sample cell line data
    cell_line_data = {
        "CELL_LINE_NAME": "A172",
        "TCGA_DESC": "GBM",
        "SITE": "central_nervous_system",
        "HISTOLOGY": "glioma",
        "GDSC_TISSUE_DESCRIPTOR_1": "nervous_system",
        "GDSC_TISSUE_DESCRIPTOR_2": "glioma",
        "CANCER_TYPE_MATCHING_TCGA_LABEL": "GBM",
        "AUC": 0.5,
        "Z_SCORE": 0.0
    }

    try:
        recommender = DrugRecommender()
        drug_count = recommender.get_drug_count()
        print(f"Drugs in database: {drug_count}")
        print(f"AUC reference rows: {len(recommender.auc_ref_df) if recommender.auc_ref_df is not None else 0}")

        if drug_count == 0:
            print("No drugs available. Skipping recommendation test.")
            return

        # Test full recommendation (top 5)
        print("\nTesting full recommendation (top 5)...")
        full_recs = recommender.recommend(cell_line_data, max_drugs=5)

        if full_recs:
            print(f"Top 5 recommendations:")
            for rec in full_recs:
                print(f"  {rec['rank']}. {rec['drug_name']}: "
                      f"LN_IC50={rec['predicted_ln_ic50']:.4f}, "
                      f"IC50={rec['predicted_ic50']:.4f}, "
                      f"Category={rec['response_category']}, "
                      f"Score%={rec['ic50_score_percentage']:.2f}%, "
                      f"Used_AUC={rec['used_auc']:.4f}")
        else:
            print("No recommendations generated")

        print("\nRecommendation test completed!")

    except Exception as e:
        print(f"Error during recommendation test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_sample_recommendation()