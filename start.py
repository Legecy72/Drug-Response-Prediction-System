"""
Startup script for the Drug Response Prediction System.

This script:
1. Checks dependencies
2. Initializes the preprocessor (fits on training data if needed)
3. Starts the FastAPI server
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if required dependencies are installed."""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
    }
    
    optional = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    optional_missing = []
    for module, package in optional.items():
        try:
            __import__(module)
        except ImportError:
            optional_missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    if optional_missing:
        print(f"WARNING: Missing optional packages: {', '.join(optional_missing)}")
        print(f"Some models may not load. Install with: pip install {' '.join(optional_missing)}")
    
    print("✓ All required dependencies are installed")
    return True


def check_data_files():
    """Check if required data files exist."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    
    required_files = [
        os.path.join(data_dir, "GDSC_DATASET.csv"),
        os.path.join(data_dir, "Compounds-annotation.csv"),
    ]
    
    missing = []
    for filepath in required_files:
        if not os.path.exists(filepath):
            missing.append(filepath)
    
    if missing:
        print(f"WARNING: Missing data files: {', '.join(missing)}")
        print("Some features may not work correctly")
    else:
        print("✓ All required data files found")
    
    # Check model files
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if model_files:
            print(f"✓ Found {len(model_files)} model files")
        else:
            print("WARNING: No model files found in model directory")
    else:
        print("WARNING: Model directory not found")
    
    return True


def initialize_preprocessor():
    """Initialize and save the preprocessor if not already saved.
    
    Uses load_and_merge_training_data() to properly merge GDSC_DATASET.csv
    with Cell_Lines_Details.xlsx (COSMIC tissue classification) to get
    SITE and HISTOLOGY columns, which are required by the preprocessor.
    """
    preprocessor_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "core", "saved_preprocessor"
    )
    
    # Check if preprocessor is already saved
    if os.path.exists(preprocessor_dir) and os.path.exists(
        os.path.join(preprocessor_dir, "scaler.pkl")
    ):
        print("✓ Preprocessor already initialized")
        return True
    
    print("Initializing preprocessor...")
    try:
        from core.preprocessing import DrugResponsePreprocessor, load_and_merge_training_data
        
        # Data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
        
        if not os.path.exists(os.path.join(data_dir, "GDSC_DATASET.csv")):
            print("WARNING: Training data not found. Preprocessor will be initialized on first request.")
            return True
        
        # Load and merge training data (includes SITE/HISTOLOGY from Excel)
        print("Loading and merging training data for preprocessor fitting...")
        df = load_and_merge_training_data(data_dir=data_dir, max_rows=50000)
        print(f"  Training data shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Initialize and fit preprocessor
        preprocessor = DrugResponsePreprocessor()
        preprocessor.fit(df)
        
        # Save preprocessor
        os.makedirs(preprocessor_dir, exist_ok=True)
        preprocessor.save(preprocessor_dir)
        
        print("✓ Preprocessor initialized and saved")
        return True
        
    except Exception as e:
        print(f"WARNING: Failed to initialize preprocessor: {e}")
        import traceback
        traceback.print_exc()
        print("Preprocessor will be initialized on first request.")
        return True


def start_server():
    """Start the FastAPI server."""
    print("\n" + "=" * 60)
    print("Starting Drug Response Prediction API Server")
    print("=" * 60)
    
    try:
        import uvicorn
        from utils.config import API_HOST, API_PORT, API_RELOAD, API_LOG_LEVEL
        
        print(f"\nServer starting at: http://{API_HOST}:{API_PORT}")
        print(f"API Documentation: http://{API_HOST}:{API_PORT}/docs")
        print(f"Frontend Application: http://{API_HOST}:{API_PORT}/app")
        print(f"\nPress Ctrl+C to stop the server\n")
        
        uvicorn.run(
            "api.main:app",
            host=API_HOST,
            port=API_PORT,
            reload=API_RELOAD,
            log_level=API_LOG_LEVEL
        )
        
    except ImportError:
        print("ERROR: uvicorn is not installed. Install with: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Drug Response Prediction System - Startup")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n[1/4] Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Check data files
    print("\n[2/4] Checking data files...")
    check_data_files()
    
    # Step 3: Initialize preprocessor
    print("\n[3/4] Initializing preprocessor...")
    initialize_preprocessor()
    
    # Step 4: Start server
    print("\n[4/4] Starting server...")
    start_server()


if __name__ == "__main__":
    main()