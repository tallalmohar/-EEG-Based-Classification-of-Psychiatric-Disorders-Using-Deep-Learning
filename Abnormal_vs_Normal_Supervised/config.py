###################
# Author - Gurleen Kaur
# File - config.py
###################

import os

# Base Paths
BASE_DATA_PATH = os.path.join("data")
BASE_REPO_PATH = os.getcwd()

# EDF Dataset Paths
TRAIN_ABNORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Abnormal", "01_tcp_ar")
TRAIN_NORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Normal", "01_tcp_ar")

# NPY Data Paths
NPY_DATA_DIR = os.path.join(BASE_DATA_PATH, "npy_data")
NPY_ABNORMAL_FOLDER = os.path.join(NPY_DATA_DIR, "abnormal")
NPY_NORMAL_FOLDER = os.path.join(NPY_DATA_DIR, "normal")

TEST_ABNORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Test_less", "Abnormal")
TEST_NORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Test_less", "Normal")

# Model Paths
MODEL_DIR = os.path.join("saved_models")
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, "eeg_classifier.pth")

# Results & Metrics Paths
RESULTS_DIR = os.path.join("XAI_Visuals")
METRICS_DIR = os.path.join(RESULTS_DIR, "Metrics")
LOSS_DIR = os.path.join(RESULTS_DIR, "Loss")

CONFUSION_MATRIX_PATH = os.path.join(METRICS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(METRICS_DIR, "roc_curve.png")
LOSS_CURVE_PATH = os.path.join(LOSS_DIR, "loss_curve.png")
ACCURACY_CURVE_PATH = os.path.join(LOSS_DIR, "accuracy_curve.png")

# XAI Paths
EEG_FILE = os.path.join("data", "Abnormal", "01_tcp_ar", "aaaaaacq_s008_t001.edf")
SALIENCY_MAP_PATH = os.path.join(RESULTS_DIR, "Saliency", "saliency.png")
FEATURE_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, "Feature_Importance", "top_features.png")

NUM_EPOCHS = 100

# Ensure required directories exist
for path in [MODEL_DIR, RESULTS_DIR, METRICS_DIR, LOSS_DIR]:
    os.makedirs(path, exist_ok=True)
