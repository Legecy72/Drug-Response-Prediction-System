# 🎓 Drug Response Prediction System — Project Briefing & Handover

> **Prepared for:** Graduation Project Discussion  
> **Role:** Lead Software Engineer  
> **Date:** May 2026

---

## 1. Project Architecture

### 1.1 System Overview

The system predicts **LN_IC50** (natural log of the half-maximal inhibitory concentration) for drug-cell line combinations using machine learning. Lower IC50 = more effective drug.

### 1.2 End-to-End Flow

```
┌─────────────┐     HTTP POST      ┌──────────────────┐
│  Web UI      │ ──────────────────► │  FastAPI Backend  │
│  (index.html)│ ◄────────────────── │  (api/main.py)    │
│  + app.js    │     JSON Response   │                    │
└─────────────┘                      └────────┬──────────┘
                                              │
                                     ┌────────▼──────────┐
                                     │   ML Service       │
                                     │  (ml_service.py)   │
                                     │                    │
                                     │ • Field Mapping    │
                                     │   snake_case →     │
                                     │   UPPER_CASE       │
                                     └────────┬──────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │      DrugResponsePredictor     │
                              │       (core/predictor.py)      │
                              │                                │
                              │  1. Validate input             │
                              │  2. Preprocess (transform)     │
                              │  3. Model.predict()            │
                              │  4. Format results             │
                              └───────┬───────────────┬────────┘
                                      │               │
                          ┌───────────▼───┐   ┌───────▼──────────┐
                          │ Preprocessor  │   │  ModelManager     │
                          │ (preprocess.) │   │ (model_manager.)  │
                          │               │   │                    │
                          │ LabelEncoder  │   │ CatBoost           │
                          │ StandardScaler│   │ XGBoost            │
                          │ (from disk!)  │   │ LightGBM           │
                          └───────────────┘   │ GradientBoosting   │
                                              │ ElasticNet         │
                                              │ Lasso, Ridge       │
                                              └────────────────────┘
```

### 1.3 Layer Responsibilities

| Layer | File | Responsibility |
|-------|------|---------------|
| **Web UI** | `web/index.html`, `web/js/app.js` | User-facing form, AJAX calls to API |
| **API Router** | `api/routers/predict.py`, `recommend.py` | HTTP endpoint definitions, request/response schemas |
| **ML Service** | `api/services/ml_service.py` | Business logic, field name mapping (snake_case → UPPER_CASE) |
| **Predictor** | `core/predictor.py` | Orchestrates: validate → preprocess → predict → format |
| **Preprocessor** | `core/preprocessing.py` | Label encoding + feature selection + StandardScaler |
| **Model Manager** | `core/model_manager.py` | Lazy model loading, prediction dispatch |
| **Recommender** | `core/recommender.py` | Iterates over drug database, predicts IC50 for each, ranks |

---

## 2. Setup & Execution

### 2.1 Prerequisites

- **Python 3.10+** (tested on 3.13)
- **pip** package manager

### 2.2 Environment Setup

```bash
# 1. Navigate to project directory
cd "e:\Graduation Project in Bioinformatics\project"

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### 2.3 Key Dependencies

```
fastapi          — Web framework
uvicorn          — ASGI server
pydantic         — Request/response validation
pandas           — Data manipulation
numpy            — Numerical operations
scikit-learn     — LabelEncoder, StandardScaler
joblib           — Model/preprocessor serialization
catboost         — CatBoost regression model
xgboost          — XGBoost regression model
lightgbm         — LightGBM regression model
openpyxl         — Excel file reading (Cell_Lines_Details.xlsx)
```

### 2.4 Starting the Server

```bash
# Option 1: Direct uvicorn
python -c "import uvicorn; uvicorn.run('api.main:app', host='127.0.0.1', port=8000, log_level='info')"

# Option 2: Using start.py (if configured)
python start.py
```

The server starts at **http://127.0.0.1:8000**.  
API docs available at **http://127.0.0.1:8000/docs** (Swagger UI).

### 2.5 Critical: The `.pkl` Artifacts

The system depends on **pre-fitted artifacts** saved to disk during the training phase:

```
model/
├── CatBoost_regression_model.pkl      # Trained CatBoost model
├── XGBoost_regression_model.pkl       # Trained XGBoost model
├── LightGBM_regression_model.pkl      # Trained LightGBM model
├── GradientBoosting_regression_model.pkl
├── ElasticNet_regression_model.pkl
├── Lasso_regression_model.pkl
└── Ridge_regression_model.pkl

core/saved_preprocessor/
├── scaler.pkl                         # Fitted StandardScaler
├── preprocessor_metadata.pkl          # is_fitted flag, feature lists
├── CELL_LINE_NAME_encoder.pkl         # Fitted LabelEncoder
├── TCGA_DESC_encoder.pkl
├── DRUG_NAME_encoder.pkl
├── TARGET_encoder.pkl
├── TARGET_PATHWAY_encoder.pkl
├── SITE_encoder.pkl
├── HISTOLOGY_encoder.pkl
├── GDSC_TISSUE_DESCRIPTOR_1_encoder.pkl
├── GDSC_TISSUE_DESCRIPTOR_2_encoder.pkl
└── CANCER_TYPE_MATCHING_TCGA_LABEL_encoder.pkl
```

**⚠️ Why this matters:** The preprocessor MUST be loaded from these files. It must NEVER be fitted at API runtime. The encoders must match exactly what was used during training — a different fit would produce different encoded values, making all predictions meaningless.

---

## 3. The Preprocessing Logic

### 3.1 Input Features (12 total)

**10 Categorical Features** (Label Encoded):

| API Field (snake_case) | Core Column (UPPER_CASE) | Example |
|------------------------|--------------------------|---------|
| `cell_line_name` | `CELL_LINE_NAME` | "A172" |
| `tcga_desc` | `TCGA_DESC` | "GBM" |
| `drug_name` | `DRUG_NAME` | "Camptothecin" |
| `target` | `TARGET` | "TOP1" |
| `target_pathway` | `TARGET_PATHWAY` | "DNA replication" |
| `site` | `SITE` | "central_nervous_system" |
| `histology` | `HISTOLOGY` | "glioma" |
| `gdsc_tissue_descriptor_1` | `GDSC_TISSUE_DESCRIPTOR_1` | "nervous_system" |
| `gdsc_tissue_descriptor_2` | `GDSC_TISSUE_DESCRIPTOR_2` | "glioma" |
| `cancer_type_matching_tcga_label` | `CANCER_TYPE_MATCHING_TCGA_LABEL` | "GBM" |

**2 Numeric Features** (Standard Scaled):

| API Field | Core Column | Range | Example |
|-----------|-------------|-------|---------|
| `auc` | `AUC` | [0.0, 1.0] | 0.93 |
| `z_score` | `Z_SCORE` | any float | 0.43 |

### 3.2 Preprocessing Pipeline (3 Steps)

```
Raw Input (12 columns)
       │
       ▼
Step 1: Label Encoding
       • Each categorical column → integer via fitted LabelEncoder
       • Unseen labels → sentinel value -1 (graceful degradation)
       • Produces 10 new _ENC columns
       │
       ▼
Step 2: Feature Selection
       • Select only the 9 "positive features" (from notebook analysis):
         AUC, Z_SCORE, TARGET_ENC, TARGET_PATHWAY_ENC, TCGA_DESC_ENC,
         GDSC_TISSUE_DESCRIPTOR_2_ENC, CANCER_TYPE_MATCHING_TCGA_LABEL_ENC,
         SITE_ENC, GDSC_TISSUE_DESCRIPTOR_1_ENC
       │
       ▼
Step 3: Standard Scaling
       • StandardScaler (fitted on training data) normalizes to mean=0, std=1
       • Output: numpy array of shape (n_samples, 9)
       │
       ▼
Model Input (9 features, scaled)
```

### 3.3 Why Pre-Fitted Artifacts Are Critical

| Scenario | What Happens | Why It's Wrong |
|----------|-------------|----------------|
| Fit preprocessor at API startup | New LabelEncoder classes may differ from training | "A172" might encode to 5 instead of 42 — model gets wrong input |
| Fit on a subset of training data | Scaler mean/std differ from what model was trained on | Scaled features have wrong distribution — predictions are garbage |
| Load from disk (correct ✅) | Exact same encoders and scaler as training | Model receives consistent input — predictions are valid |

**The golden rule:** The preprocessor is part of the model. Changing it is like swapping out a layer of a neural network and expecting the same output.

---

## 4. Model Ensemble

### 4.1 Available Models (7 total)

| Model | Type | Filename |
|-------|------|----------|
| **CatBoost** | Gradient boosting (default) | `CatBoost_regression_model.pkl` |
| **XGBoost** | Gradient boosting | `XGBoost_regression_model.pkl` |
| **LightGBM** | Gradient boosting | `LightGBM_regression_model.pkl` |
| **GradientBoosting** | sklearn gradient boosting | `GradientBoosting_regression_model.pkl` |
| **ElasticNet** | Regularized linear | `ElasticNet_regression_model.pkl` |
| **Lasso** | L1 regularized linear | `Lasso_regression_model.pkl` |
| **Ridge** | L2 regularized linear | `Ridge_regression_model.pkl` |

### 4.2 Model Selection Logic

```
1. User specifies model_name in request?
   └─ YES → Use that model
   └─ NO  → Use default model (CatBoost)

2. Default model not available?
   └─ Fall back to first discovered model

3. Model not yet loaded in memory?
   └─ Lazy-load from disk via joblib.load()

4. All models take same input: (n_samples, 9) scaled features
   All models output: (n_samples,) predicted LN_IC50
```

### 4.3 Why CatBoost as Default?

CatBoost typically performs best on categorical-heavy tabular data. It handles encoded categorical features well and was likely the top performer during model evaluation in the notebook.

---

## 5. The "Critical Bug" Story

### 5.1 The Bug

**Error message:** `"Preprocessor must be fitted before transformation"`

This error appeared at **prediction time** — meaning the API would start successfully but crash when a user tried to get a prediction.

### 5.2 Root Causes (3 interrelated)

```
Cause 1: Silent Fallback
┌─────────────────────────────────────────────────────────┐
│ _load_or_create_preprocessor() in predictor.py          │
│                                                         │
│ if saved_preprocessor_exists:                           │
│     return load_it()                                    │
│ else:                                                   │
│     return DrugResponsePreprocessor()  ← UNFITTED! ❌   │
└─────────────────────────────────────────────────────────┘

Cause 2: Runtime Fitting in Lifespan Handler
┌─────────────────────────────────────────────────────────┐
│ api/main.py lifespan handler                            │
│                                                         │
│ if not scaler.pkl exists:                               │
│     df = load_training_data()                           │
│     preprocessor = DrugResponsePreprocessor()           │
│     preprocessor.fit(df)  ← FITS AT RUNTIME! ❌         │
│     preprocessor.save(...)                              │
└─────────────────────────────────────────────────────────┘

Cause 3: No Validation After Loading
┌─────────────────────────────────────────────────────────┐
│ If saved artifacts were corrupted:                      │
│   preprocessor.is_fitted could be False                 │
│   But no check was made until prediction time           │
│   → Error surfaces in user's request, not at startup ❌ │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Why This Was Dangerous

| Aspect | Impact |
|--------|--------|
| **Silent failure** | API starts "successfully" but is broken — no error until first user request |
| **Data leakage risk** | Fitting on full training data at runtime could include test data |
| **Inconsistent encoding** | If fit on different data, same category maps to different integer — model gets garbage |
| **Production crash** | Any transient file system issue could trigger the unfitted fallback |

### 5.4 The Fix

```
BEFORE (broken):
                    ┌──────────────────┐
                    │ Load preprocessor │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Exists on disk?   │
                    └──┬────────────┬───┘
                   YES │            │ NO
                    ┌──▼───┐   ┌───▼──────────────┐
                    │ Load │   │ Create UNFITTED   │
                    │      │   │ preprocessor ❌    │
                    └──────┘   └───────────────────┘

AFTER (fixed):
                    ┌──────────────────┐
                    │ Load preprocessor │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Exists on disk?   │
                    └──┬────────────┬───┘
                   YES │            │ NO
                    ┌──▼───┐   ┌───▼──────────────┐
                    │ Load │   │ RuntimeError ✅    │
                    │      │   │ "Preprocessor not  │
                    └──┬───┘   │ found — cannot     │
                       │       │ proceed"           │
                ┌──────▼────┐  └───────────────────┘
                │ is_fitted? │
                └──┬─────┬──┘
                YES│     │NO
                   │  ┌──▼──────────────┐
                   │  │ RuntimeError ✅  │
                   │  │ "Corrupted       │
                   │  │ artifacts"       │
                   │  └─────────────────┘
                ┌──▼───┐
                │  OK  │
                └──────┘
```

**Key principle:** Fail fast, fail loud. If the preprocessor isn't available, the service should refuse to start — not silently degrade.

### 5.5 Logging Architecture Added

| Level | What's Logged | Purpose |
|-------|--------------|---------|
| **INFO** | API request/response summaries, predictor input shape, startup status | Production monitoring |
| **DEBUG** | Per-sample details, transform stats, model internals | Development debugging |
| **ERROR** | Missing preprocessor, validation failures, prediction errors | Alert on failures |

Example log output for a prediction:
```
INFO  [API] Prediction request: cell_line=A172, drug=Camptothecin, target=TOP1, model=None
INFO  [PREDICT] Input features: shape=(1, 12), columns=[...], sample_count=1
INFO  [API] Prediction result: ic50=-3.0533, model=CatBoost
```

---

## 6. Technical Specifications

### 6.1 API Endpoints

| Method | Endpoint | Description | Request Body |
|--------|----------|-------------|-------------|
| `POST` | `/predict/single` | Single IC50 prediction | `PredictionRequest` |
| `POST` | `/predict/batch` | Batch predictions | `BatchPredictionRequest` |
| `POST` | `/predict/compare` | Compare models | `ModelComparisonRequest` |
| `POST` | `/recommend` | Drug recommendations | `RecommendationRequest` |
| `GET` | `/models` | List available models | — |
| `GET` | `/health` | Health check | — |
| `GET` | `/docs` | Swagger API docs | — |
| `GET` | `/` | API info | — |
| `GET` | `/app` | Web UI | — |

### 6.2 Request Schema (PredictionRequest)

```json
{
  "cell_line_name": "A172",
  "tcga_desc": "GBM",
  "site": "central_nervous_system",
  "histology": "glioma",
  "gdsc_tissue_descriptor_1": "nervous_system",
  "gdsc_tissue_descriptor_2": "glioma",
  "cancer_type_matching_tcga_label": "GBM",
  "drug_name": "Camptothecin",
  "target": "TOP1",
  "target_pathway": "DNA replication",
  "auc": 0.93,
  "z_score": 0.43,
  "model_name": null
}
```

### 6.3 Response Schema (PredictionResponse)

```json
{
  "success": true,
  "prediction": {
    "predicted_ic50": -3.0533,
    "model_used": "CatBoost"
  },
  "timestamp": "2026-05-01T18:39:44.086985"
}
```

### 6.4 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | Latest |
| ASGI Server | Uvicorn | Latest |
| Data Validation | Pydantic v2 | Latest |
| Data Processing | Pandas, NumPy | Latest |
| ML Preprocessing | scikit-learn | Latest |
| ML Models | CatBoost, XGBoost, LightGBM, sklearn | Latest |
| Serialization | joblib | Latest |
| Excel Reading | openpyxl | Latest |
| Containerization | Docker | — |

### 6.5 Project Structure

```
project/
├── api/                          # FastAPI application layer
│   ├── main.py                   # App entry point, lifespan, CORS
│   ├── routers/                  # Endpoint definitions
│   │   ├── predict.py            # /predict/* endpoints
│   │   ├── recommend.py          # /recommend endpoint
│   │   ├── models.py             # /models endpoint
│   │   └── health.py             # /health endpoint
│   ├── schemas/                  # Pydantic request/response models
│   │   ├── request.py            # Input validation schemas
│   │   └── response.py           # Output format schemas
│   └── services/
│       └── ml_service.py         # Business logic, field mapping
├── core/                         # ML engine layer
│   ├── predictor.py              # Prediction orchestrator
│   ├── preprocessing.py          # LabelEncoder + StandardScaler pipeline
│   ├── model_manager.py          # Model discovery, loading, prediction
│   ├── recommender.py            # Drug ranking engine
│   └── saved_preprocessor/       # Pre-fitted preprocessor artifacts
│       ├── scaler.pkl
│       ├── preprocessor_metadata.pkl
│       └── *_encoder.pkl (10 files)
├── model/                        # Trained model files (7 .pkl)
├── data/                         # Training datasets
│   ├── GDSC_DATASET.csv
│   ├── GDSC2-dataset.csv
│   ├── Compounds-annotation.csv
│   └── Cell_Lines_Details.xlsx
├── web/                          # Frontend
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── utils/                        # Configuration & logging
│   ├── config.py
│   └── logger.py
├── notebooks/                    # Jupyter notebooks (training)
├── tests/                        # Unit tests
├── requirements.txt
├── Dockerfile
└── start.py
```

---

## 7. Quick Reference for Discussion

### Q: "How does a prediction flow through the system?"

> User submits JSON → FastAPI validates with Pydantic → MLService maps snake_case to UPPER_CASE → Predictor validates 12 required columns → Preprocessor transforms (LabelEncode 10 categoricals → select 9 features → StandardScale) → Model predicts → Result formatted and returned.

### Q: "Why must the preprocessor be pre-fitted?"

> The LabelEncoders assign integer IDs based on the order classes appear in training data. If we re-fit, "A172" might encode to a different number. The StandardScaler's mean/std come from training data. If different, the scaled values shift. Both would make the model receive input it never saw during training — garbage in, garbage out.

### Q: "What was the biggest technical challenge?"

> The "Unfitted Preprocessor" bug. The system would silently create an unfitted preprocessor when saved artifacts weren't found, then crash at prediction time with a confusing error. The fix was threefold: (1) raise RuntimeError immediately if preprocessor can't be loaded, (2) validate is_fitted after loading, (3) remove the lifespan handler's runtime fitting code. We also added structured logging so issues are visible at startup, not at prediction time.

### Q: "How do you handle unseen categories?"

> Graceful degradation: unseen labels are mapped to sentinel value -1. This allows the system to still make predictions for new cell lines or drugs, rather than crashing. The model learns to handle the -1 encoding from training data where it wasn't present, so predictions for unseen categories are less reliable but still produced.

### Q: "Why 9 features specifically?"

> From the notebook's feature importance analysis, these 9 features had positive importance scores: 2 numeric (AUC, Z_SCORE) + 7 encoded categoricals (TARGET_ENC, TARGET_PATHWAY_ENC, TCGA_DESC_ENC, GDSC_TISSUE_DESCRIPTOR_2_ENC, CANCER_TYPE_MATCHING_TCGA_LABEL_ENC, SITE_ENC, GDSC_TISSUE_DESCRIPTOR_1_ENC). Note: not all 10 categoricals are used — DRUG_NAME_ENC and CELL_LINE_NAME_ENC were excluded by the feature selection process.

---

*End of Project Briefing & Handover*
