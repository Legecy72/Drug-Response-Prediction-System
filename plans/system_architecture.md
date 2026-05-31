# System Architecture

## Overall System Flow

```mermaid
flowchart TD
    A[User Input] --> B[Web Application UI]
    B --> C[Backend API]
    C --> D[Preprocessing Module]
    D --> E[Model Manager]
    E --> F[ML Model Prediction]
    F --> G[IC50 Prediction]
    G --> H[Drug Recommendation Engine]
    H --> I[Ranked Drug List]
    I --> J[Results Display]
```

## Component Architecture

```mermaid
graph TB
    subgraph Frontend
        A[Model Selection] --> B[Feature Input Form]
        B --> C[Prediction Display]
        B --> D[Recommendation Table]
        C --> E[Visualization Charts]
    end

    subgraph Backend
        F[FastAPI Server] --> G[Predict Router]
        F --> H[Recommend Router]
        F --> I[Models Router]
        
        G --> J[ML Service]
        H --> J
        
        J --> K[Preprocessing Service]
        J --> L[Model Manager]
        J --> M[Recommender Service]
        
        K --> N[Label Encoders]
        K --> O[Standard Scaler]
        
        L --> P[Model Registry<br/>CatBoost, XGBoost, etc.]
        
        M --> Q[Drug Database]
    end

    subgraph Data Layer
        R[Trained Models .pkl]
        S[GDSC Dataset]
        T[Compounds Annotation]
    end

    P --> R
    Q --> S
    Q --> T
```

## ML Pipeline Sequence

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Preprocessor
    participant Model
    participant Recommender

    User->>Frontend: Submit cell line + drug features
    Frontend->>Backend: POST /predict
    Backend->>Preprocessor: Preprocess input
    Preprocessor->>Preprocessor: Label encoding
    Preprocessor->>Preprocessor: Feature scaling
    Preprocessor->>Model: Prepare features
    Model->>Model: Predict IC50
    Model->>Backend: Return prediction
    Backend->>Frontend: JSON response
    
    User->>Frontend: Request drug recommendations
    Frontend->>Backend: POST /recommend
    Backend->>Recommender: Get all drugs
    loop For each drug
        Recommender->>Preprocessor: Preprocess drug
        Recommender->>Model: Predict IC50
        Recommender->>Recommender: Calculate score
    end
    Recommender->>Recommender: Rank drugs
    Recommender->>Backend: Return ranked list
    Backend->>Frontend: JSON response
```

## Data Flow

```mermaid
flowchart LR
    A[Raw Input Features] --> B[Data Validation]
    B --> C[Missing Value Handling]
    C --> D[Categorical Encoding]
    D --> E[Feature Scaling]
    E --> F[Feature Selection<br/>9 features]
    F --> G[Model Prediction]
    G --> H[IC50 Output]
    H --> I[Score Calculation]
    I --> J[Ranking Algorithm]
    J --> K[Final Recommendations]
```

## Deployment Architecture

```mermaid
graph TB
    subgraph Production
        A[Load Balancer] --> B[API Instance 1]
        A --> C[API Instance 2]
        A --> D[API Instance N]
        
        B --> E[Shared Model Storage]
        C --> E
        D --> E
        
        F[Static File Server] --> G[Frontend Assets]
        
        H[User Browser] --> A
        H --> F
    end
    
    subgraph Development
        I[Local Development] --> J[Docker Compose]
        J --> K[API Container]
        J --> L[Frontend Container]
    end
```

## Key Components Description

### 1. Preprocessing Module
- **Input**: Raw feature values (categorical + numeric)
- **Processing**: 
  - Label encoding for 10 categorical features
  - Standard scaling for all 9 features
  - Feature selection to match training set
- **Output**: Processed features ready for model prediction

### 2. Model Manager
- **Discovery**: Auto-detects `.pkl` files in `/model/`
- **Loading**: Lazy loading of models on first use
- **Caching**: Keeps loaded models in memory
- **Interface**: Uniform prediction interface across all models

### 3. Recommendation Engine
- **Drug Database**: All available drugs from Compounds-annotation.csv
- **Scoring**: `score = 1 / (1 + IC50)`
- **Ranking**: Sort by score (descending)
- **Filtering**: Optional by target pathway, drug class

### 4. API Layer
- **FastAPI**: Modern, fast web framework
- **Validation**: Pydantic models for request/response
- **Documentation**: Auto-generated OpenAPI/Swagger UI
- **Error Handling**: Comprehensive error responses

### 5. Frontend Application
- **Responsive Design**: Works on desktop and mobile
- **Dynamic Forms**: Adapts to selected model features
- **Visualization**: Charts for scores and rankings
- **User Experience**: Intuitive workflow for researchers

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI (Python) | REST API server |
| ML Framework | Scikit-learn, XGBoost, etc. | Model inference |
| Frontend | React/JavaScript | User interface |
| Data Processing | Pandas, NumPy | Feature engineering |
| Model Serialization | Joblib | .pkl file loading |
| Containerization | Docker | Deployment |
| Testing | Pytest | Quality assurance |

## Next Steps
1. Review and approve architecture
2. Begin implementation with Phase 2 (ML Engine)
3. Validate each component against notebook results
4. Iterate through remaining phases