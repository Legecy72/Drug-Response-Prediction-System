# Drug Response Prediction System

## Overview

Drug Response Prediction System is a Bioinformatics graduation project that predicts cancer cell line sensitivity to anti-cancer drugs using Machine Learning models.

The system integrates:

- Drug Response Prediction
- Drug Recommendation Engine
- Interactive AI Chatbot Assistant
- FastAPI Backend
- Modern Web Interface

---

## Features

### Prediction Module

Predicts LN_IC50 values using trained regression models:

- CatBoost
- XGBoost
- LightGBM
- Gradient Boosting
- ElasticNet
- Ridge
- Lasso

---

### Recommendation Engine

Ranks candidate drugs based on:

- Predicted LN_IC50
- Predicted IC50
- IC50 Score
- Cancer context matching

---

### AI Chatbot Assistant

Provides:

- IC50 interpretation
- Drug sensitivity explanations
- Recommendation explanations
- Model information
- Bioinformatics guidance

Powered by:

- Google Gemini API
- Fallback Response Engine

---

## Technology Stack

### Backend

- Python
- FastAPI
- Pydantic
- Scikit-Learn
- CatBoost
- XGBoost
- LightGBM

### Frontend

- HTML
- CSS
- JavaScript

### AI

- Google Gemini API

---

## Project Structure

```text
project/
│
├── api/
├── core/
├── data/
├── models/
├── web/
├── scripts/
├── docs/
├── tests/
│
├── start.py
├── requirements.txt
└── README.md
```

---

## Running the Project

### Create Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Application

```bash
python start.py
```

---

## Application URLs

Frontend:

```text
http://localhost:8000/app
```

API Docs:

```text
http://localhost:8000/docs
```

Health Check:

```text
http://localhost:8000/health
```

---

## Authors

Graduation Project in Bioinformatics

Faculty of Artificial Intelligence Delta univ

2026
