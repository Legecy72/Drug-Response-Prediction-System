# Comprehensive Plan: Production-Ready Drug Response Prediction System

## Project Overview
This project transforms an existing bioinformatics graduation project into a complete, production-ready system for predicting drug response in cancer cell lines using machine learning. The system will include a full ML pipeline, backend API, web application, and drug recommendation engine.

## Key Findings from Phase 1 Analysis

### Data Sources
1. **GDSC Dataset** (`GDSC_DATASET.csv`, `GDSC2-dataset.csv`) - Primary drug response data
2. **Cell Lines Details** (`Cell_Lines_Details.xlsx`) - Cell line metadata
3. **Compounds Annotation** (`Compounds-annotation.csv`) - Drug target and pathway information

### Target Variable
- **Primary target**: `LN_IC50` (natural log of half-maximal inhibitory concentration)
- **Secondary targets**: `AUC` (Area Under Curve), `Z_SCORE`

### Features Used in Training
Based on notebook analysis, the final feature set includes 9 features:

1. **AUC** (numeric)
2. **Z_SCORE** (numeric)
3. **TARGET_ENC** (encoded drug target)
4. **TARGET_PATHWAY_ENC** (encoded target pathway)
5. **TCGA_DESC_ENC** (encoded TCGA description)
6. **GDSC_TISSUE_DESCRIPTOR_2_ENC** (encoded tissue descriptor 2)
7. **CANCER_TYPE_MATCHING_TCGA_LABEL_ENC** (encoded cancer type)
8. **SITE_ENC** (encoded site)
9. **GDSC_TISSUE_DESCRIPTOR_1_ENC** (encoded tissue descriptor 1)

### Categorical Columns (Label Encoded)
- `CELL_LINE_NAME`, `TCGA_DESC`, `DRUG_NAME`, `TARGET`, `TARGET_PATHWAY`, `SITE`, `HISTOLOGY`, `GDSC_TISSUE_DESCRIPTOR_1`, `GDSC_TISSUE_DESCRIPTOR_2`, `CANCER_TYPE_MATCHING_TCGA_LABEL`

### Preprocessing Pipeline
1. **Data Cleaning**: Column name standardization, handling missing values
2. **Encoding**: Label encoding for all categorical variables (10 columns)
3. **Feature Selection**: Correlation-based selection to 9 positive features
4. **Scaling**: StandardScaler applied to all features
5. **Train/Test Split**: 80/20 split with random_state=42

### Trained Models (Saved in `/model/`)
- CatBoost Regression
- ElasticNet Regression
- Gradient Boosting Regression
- Lasso Regression
- LightGBM Regression
- Ridge Regression
- XGBoost Regression

### Input/Output Structure
- **Input**: Cell line features + drug features
- **Output**: Predicted LN_IC50 value (continuous)
- **Recommendation**: Score = 1 / (1 + IC50), lower IC50 = better

## Phase 2: Production ML Engine

### 2.1 Preprocessing Module
- Create `preprocessing.py` with exact replication of notebook preprocessing
- Implement `LabelEncoder` persistence for categorical features
- Save `StandardScaler` fitted on training data
- Ensure feature alignment (9 features in exact order)

### 2.2 Model Manager
- Create `model_manager.py` to auto-detect models in `/model/`
- Implement dynamic loading of `.pkl` files
- Support multiple model selection via API
- Validate input features match training features

### 2.3 Prediction Pipeline
- Build `predictor.py` with end-to-end pipeline:
  - Input validation → preprocessing → scaling → prediction → output
- Handle batch predictions for drug recommendation

## Phase 3: Drug Recommendation Engine

### 3.1 Recommendation Logic
- For given cell line, evaluate all drugs (or filtered subset)
- Predict IC50 for each drug using selected model
- Compute score: `score = 1 / (1 + IC50)`
- Rank drugs by score (higher = better)

### 3.2 Optimization
- Cache drug list and precomputed features
- Implement batch prediction for performance
- Add filtering by drug class/target pathway

### 3.3 Output Format
- Structured JSON with: drug name, IC50, score, ranking
- Optional: confidence intervals, alternative models

## Phase 4: Backend (FastAPI)

### 4.1 API Structure
```
/api
  /routers
    - predict.py
    - recommend.py
    - models.py
  /services
    - ml_service.py
    - preprocessing_service.py
  /schemas
    - request.py
    - response.py
  /models
    - database models (if needed)
```

### 4.2 Endpoints
- `POST /predict` - Single prediction
- `POST /recommend` - Drug recommendations for cell line
- `GET /models` - List available models
- `GET /health` - Health check

### 4.3 Features
- Pydantic validation for all inputs
- Comprehensive error handling
- Request/response logging
- CORS configuration
- API documentation (Swagger/OpenAPI)

## Phase 5: Full Web Application

### 5.1 Frontend Architecture
- React/Vue.js or simple HTML/CSS/JS
- Clean, professional UI suitable for academic presentation
- Responsive design

### 5.2 Core Pages/Components
1. **Model Selection** - Dropdown to choose ML model
2. **Feature Input Form** - Dynamic form based on 9 features
3. **Prediction View** - Display IC50 with interpretation
4. **Recommendation Table** - Sortable table of ranked drugs
5. **Visualization** - Score distribution, drug effectiveness charts

### 5.3 Key Features
- Real-time validation
- Example data/pre-filled forms
- Export results (CSV/PDF)
- Historical prediction tracking (if backend supports)

## Phase 6: System Integration

### 6.1 Frontend-Backend Connection
- API client implementation
- Error handling and loading states
- Environment configuration

### 6.2 End-to-End Testing
- Test prediction flow
- Test recommendation flow
- Error scenario handling

### 6.3 Deployment Configuration
- Docker setup for backend
- Static frontend hosting
- Environment variables

## Phase 7: Testing & Validation

### 7.1 Validation Requirements
- Ensure predictions match notebook results (sample validation)
- Test edge cases (missing values, extreme inputs)
- Performance testing (response times)

### 7.2 Test Suite
- Unit tests for preprocessing
- Integration tests for API endpoints
- Frontend component tests

### 7.3 Consistency Checks
- Feature alignment verification
- Model loading reliability
- Data pipeline reproducibility

## Phase 8: Final Structure

### 8.1 Project Organization
```
/project
  /data                    # Original datasets
  /model                   # Trained models (.pkl)
  /api                     # FastAPI backend
    /routers
    /services
    /schemas
    main.py
  /frontend                # Web application
    /public
    /src
  /core                    # ML core modules
    preprocessing.py
    model_manager.py
    predictor.py
    recommender.py
  /utils                   # Utilities
    logger.py
    config.py
  /tests                   # Test suite
  /docs                    # Documentation
  /plans                   # Project plans
  requirements.txt
  docker-compose.yml
  README.md
```

### 8.2 Documentation
- Setup instructions
- API documentation
- User guide
- Technical architecture

### 8.3 Deployment
- Local development setup
- Production deployment guide
- Environment configuration

## Technical Considerations

### Data Consistency
- Must exactly replicate notebook preprocessing
- Label encoders must be saved and reused
- Feature order must match training data

### Performance
- Model loading should be lazy/cached
- Batch predictions for recommendation
- Consider async endpoints for long operations

### Scalability
- Stateless design for horizontal scaling
- Database optional for prediction history
- File-based model storage (can transition to model registry)

### Security
- Input validation/sanitization
- Rate limiting (if public)
- No sensitive data in this project

## Next Steps
1. **Approval**: Review this plan and confirm approach
2. **Implementation**: Begin with Phase 2 (Production ML Engine)
3. **Iteration**: Complete phases sequentially with validation at each step

## Dependencies
- Python 3.8+
- FastAPI, Uvicorn
- Scikit-learn, Pandas, NumPy
- Joblib (for model loading)
- Frontend: React/Vue.js or vanilla JS
- Testing: Pytest

## Success Criteria
- Fully working backend API with prediction/recommendation
- Complete web application with intuitive UI
- ML pipeline that reproduces notebook results
- Clean, documented, maintainable code
- Academic presentation-ready system