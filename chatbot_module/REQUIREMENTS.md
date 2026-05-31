# Python Dependencies — Chatbot Module

## Required Packages

These are the Python packages required for the chatbot module to function.

### Core Chatbot Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| `google-generativeai` | >=0.3.0 | Gemini API client for AI-powered responses |
| `python-dotenv` | >=1.0.0 | Load environment variables from `.env` file |
| `fastapi` | >=0.95.0 | Web framework for API endpoints |
| `uvicorn[standard]` | >=0.21.0 | ASGI server to run FastAPI |
| `pydantic` | >=1.10.0 | Request/response validation schemas |

### Already in Main Project (Not Needed Separately)

These packages are required by the main project but are NOT specific to the chatbot:

| Package | Version | Purpose |
|---|---|---|
| `pandas` | >=1.5.0 | Data manipulation (ML service) |
| `numpy` | >=1.23.0 | Numerical operations (ML service) |
| `scikit-learn` | >=1.2.0 | Preprocessing pipeline (ML service) |
| `joblib` | >=1.2.0 | Model/preprocessor loading (ML service) |
| `openpyxl` | >=1.1.0 | Excel file reading (data loading) |
| `xgboost` | >=1.7.0 | ML model |
| `lightgbm` | >=3.3.0 | ML model |
| `catboost` | >=1.1.0 | ML model |

### Testing Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| `pytest` | >=7.3.0 | Test runner |
| `pytest-cov` | >=4.0.0 | Test coverage reporting |
| `httpx` | >=0.24.0 | HTTP client for testing FastAPI |

## Installation

### Install Only Chatbot Dependencies

```bash
pip install google-generativeai>=0.3.0 python-dotenv>=1.0.0 fastapi>=0.95.0 "uvicorn[standard]>=0.21.0" pydantic>=1.10.0
```

### Install All Project Dependencies (Including Chatbot)

```bash
pip install -r requirements.txt
```

### Install Testing Dependencies

```bash
pip install pytest>=7.3.0 pytest-cov>=4.0.0 httpx>=0.24.0
```

## Version Notes

- **Pydantic V2**: The schemas use `json_schema_extra` (V2 syntax). If you're on Pydantic V1, change to `schema_extra`.
- **google-generativeai**: Version 0.3.0+ includes the `GenerativeModel` with `system_instruction` support.
- **uvicorn[standard]**: The `[standard]` extra includes `uvloop` and `httptools` for better performance.