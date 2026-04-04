"""
Universal Data Science Pipeline - Configuration
================================================
All settings, flags, and paths are controlled from here.
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_PREP_DIR = os.path.join(OUTPUT_DIR, "data_preparation")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "feature_analysis")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model subdirectories
REGRESSION_DIR = os.path.join(MODELS_DIR, "regression")
CLASSIFICATION_DIR = os.path.join(MODELS_DIR, "classification")
CLUSTERING_DIR = os.path.join(MODELS_DIR, "clustering")
TIMESERIES_DIR = os.path.join(MODELS_DIR, "time_series")
ANOMALY_DIR = os.path.join(MODELS_DIR, "anomaly_detection")
ASSOCIATION_DIR = os.path.join(MODELS_DIR, "association_rules")
ENSEMBLE_DIR = os.path.join(MODELS_DIR, "hybrid_ensemble")

# ============================================================
# FLAGS
# ============================================================
ENABLE_DEEP_LEARNING = False  # Toggle: True to include DL models
ENABLE_TIME_SERIES = True     # Auto-detected, but can force off
ENABLE_ASSOCIATION = True     # Market basket analysis
ENABLE_ANOMALY = True         # Anomaly detection phase
ENABLE_ENSEMBLE = True        # Hybrid/ensemble models

# ============================================================
# DATA SETTINGS
# ============================================================
MAX_ROWS_FOR_GP = 5000        # Gaussian Process row limit
MAX_CATEGORIES_ONEHOT = 15    # Max unique values before switching to label encoding
MISSING_THRESHOLD = 0.7       # Drop column if >70% missing
OUTLIER_METHOD = "iqr"        # "iqr" or "zscore"
SCALING_METHOD = "standard"   # "standard", "minmax", or "robust"
TEST_SIZE = 0.2               # Train/test split ratio
RANDOM_STATE = 42             # Reproducibility seed

# ============================================================
# DEEP LEARNING SETTINGS
# ============================================================
DL_EPOCHS = 100
DL_BATCH_SIZE = 32
DL_EARLY_STOPPING_PATIENCE = 10
DL_LEARNING_RATE = 0.001
DL_VALIDATION_SPLIT = 0.2

# ============================================================
# TIME SERIES SETTINGS
# ============================================================
TS_FORECAST_HORIZON = 10      # Steps to forecast
TS_WINDOW_SIZE = 10           # Lookback window for LSTM/GRU

# ============================================================
# LOGGING
# ============================================================
LOG_FILE = os.path.join(LOGS_DIR, "pipeline_log.txt")
VERBOSE = True                # Print progress to console