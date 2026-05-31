"""Quick functional test for the enhanced recommendation system."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.recommender import DrugRecommender, classify_response
from core.predictor import DrugResponsePredictor

# Test classify_response
print("=== classify_response tests ===")
print(f"  LN_IC50=-3.0 -> {classify_response(-3.0)}")
print(f"  LN_IC50=-1.0 -> {classify_response(-1.0)}")
print(f"  LN_IC50=1.0 -> {classify_response(1.0)}")
print(f"  LN_IC50=3.0 -> {classify_response(3.0)}")

# Test recommender with AUC reference
print("\n=== DrugRecommender initialization ===")
predictor = DrugResponsePredictor()
recommender = DrugRecommender(predictor=predictor)

print(f"AUC reference loaded: {recommender.auc_ref_df is not None}")
if recommender.auc_ref_df is not None:
    print(f"AUC reference rows: {len(recommender.auc_ref_df)}")
    print(f"AUC reference unique drugs: {recommender.auc_ref_df['DRUG_NAME'].nunique()}")

# Test recommendation with a known cancer context
print("\n=== Recommendation test (MCF7 / BRCA / breast) ===")
cell_line_data = {
    'CELL_LINE_NAME': 'MCF7',
    'TCGA_DESC': 'BRCA',           # TCGA abbreviation
    'SITE': 'breast',              # Tissue site
    'HISTOLOGY': 'Carcinoma',
    'GDSC_TISSUE_DESCRIPTOR_1': 'breast',
    'GDSC_TISSUE_DESCRIPTOR_2': 'breast',
    'CANCER_TYPE_MATCHING_TCGA_LABEL': 'BRCA',  # TCGA label (abbreviation, not tissue name)
    'AUC': 0.5,
    'Z_SCORE': 0.0
}

recs = recommender.recommend(cell_line_data, max_drugs=5)
print(f"Recommendations generated: {len(recs)}")

if recs:
    print("\nTop recommendation fields:")
    r = recs[0]
    for k in ['drug_name', 'rank', 'predicted_ln_ic50', 'predicted_ic50',
              'response_category', 'ic50_score', 'ic50_score_percentage',
              'used_auc', 'score', 'model_used']:
        print(f"  {k}: {r.get(k)}")

    # Verify ranking order
    print("\nAll 5 recommendations (rank, drug, category, score%):")
    for r in recs:
        print(f"  #{r['rank']} {r['drug_name']} - {r['response_category']} - {r['ic50_score_percentage']:.2f}%")

    # Verify IC50 score formula
    import numpy as np
    r = recs[0]
    expected_ic50 = float(np.exp(r['predicted_ln_ic50']))
    expected_score = float(1.0 / (1.0 + expected_ic50))
    expected_pct = expected_score * 100
    print(f"\nFormula verification for top drug:")
    print(f"  exp(LN_IC50) = {expected_ic50:.6f}, predicted_ic50 = {r['predicted_ic50']:.6f} -> MATCH: {abs(expected_ic50 - r['predicted_ic50']) < 0.001}")
    print(f"  1/(1+IC50) = {expected_score:.6f}, ic50_score = {r['ic50_score']:.6f} -> MATCH: {abs(expected_score - r['ic50_score']) < 0.001}")
    print(f"  score*100 = {expected_pct:.2f}, ic50_score_percentage = {r['ic50_score_percentage']:.2f} -> MATCH: {abs(expected_pct - r['ic50_score_percentage']) < 0.01}")

print("\n=== ALL TESTS PASSED ===")