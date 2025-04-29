###################
# Author - Gurleen Kaur, Edited by Michaela Foo
# File - config.py
###################

import os

# EDF Dataset Paths
BASE_DATA_PATH = r"dataset"

# NPY Data Paths
NPY_DATA_DIR = os.path.join(BASE_DATA_PATH, "npy_data")
NPY_ABNORMAL_FOLDER = os.path.join(NPY_DATA_DIR, "abnormal")
NPY_NORMAL_FOLDER = os.path.join(NPY_DATA_DIR, "normal")

TEST_ABNORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Test", "Abnormal")
TEST_NORMAL_FOLDER = os.path.join(BASE_DATA_PATH, "Test", "Normal")

# Base Repo Path
BASE_REPO_PATH = r"abnormal_vs_norm_SSupervised/"

# Model Paths
MODEL_DIR = os.path.join(BASE_REPO_PATH, "saved_models")
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, "eeg_classifier.pth")

LOSS_CURVE_PATH = "loss_curve.png"
ACCURACY_CURVE_PATH = "accuracy_curve.png"


