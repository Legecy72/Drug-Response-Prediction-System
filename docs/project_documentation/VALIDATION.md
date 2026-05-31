# Validation Notes

## Checked Inputs

- Regression package artifacts were copied into `backend/artifacts/regression`.
- Classification package artifacts were copied into `backend/artifacts/classification`.
- Chatbot package was copied into `backend/chatbot_module`.
- Notebook logic was reviewed for artifact creation and preprocessing order.

## Model Defaults

- Regression default: `lightgbm`
  - Best R2 in `regression_results.csv`: `0.9406480479341951`
- Classification default: `xgboost`
  - Best accuracy/F1 in `classification_results.csv`: accuracy `0.8550385599016673`, F1 `0.8182450907283122`

## Fixes Applied

- The provided `classification_model_engine.py` loaded `classification_encoder.pkl` but did not apply it before prediction.
- Live validation showed `classification_encoder.pkl` was fitted with `0` input features, so it cannot encode the string web inputs.
- The new engine falls back to `regression_label_encoders.pkl` for the shared categorical columns, then applies the numeric imputer and scaler before prediction.
- Recommendation now returns all matching drugs by default. `top_n` is optional.
- Recommendation sorting uses:
  - Sensitive before Resistant
  - Lower predicted IC50 first
  - Higher sensitivity probability as tie-breaker

## Runtime Validation

The local Codex runtime did not include ML dependencies initially. They were installed temporarily into `work/pydeps` for validation. The important pickle compatibility package, `scikit-learn`, was aligned to `1.6.1`.

Syntax validation was completed successfully with Python AST parsing.

Live validation succeeded:

- `GET /api/health` returned `ok`.
- `GET /api/status` loaded 241,993 AUC reference rows and 286 drugs.
- `POST /api/predict` succeeded for the sample input:
  - Drug: `Camptothecin`
  - Predicted IC50: `0.02624814203975672`
  - Sensitivity: `Sensitive`
- `POST /api/recommend` succeeded for top 3 recommendations:
  - First ranked drug: `SN-38`

The chatbot endpoint also loaded in fallback mode because `GEMINI_API_KEY` was not configured.

Port `8000` was already in use on this machine, so the validation server was run on `http://127.0.0.1:8001`.

## Data Limitation

The regression pipeline requires an exact matching row in `drug_auc_reference.csv` to obtain AUC. If a user chooses a combination that does not exist in that CSV, the API returns a clear error instead of guessing.
