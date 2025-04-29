###################
# Author - Gurleen Kaur
# File - run_all.py
###################

import sys
import os

def check_requirements():
    """
    Check that all required libraries are installed.
    If some packages are missing, list them and exit.
    """
    required_packages = [
        "torch", "numpy", "matplotlib", "sklearn",
        "tqdm", "tensorboard"
    ]
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    if missing_packages:
        print("[ERROR] Missing required packages: " + ", ".join(missing_packages))
        print("Please install them using:\n\n  pip install " + " ".join(missing_packages))
        return False
    else:
        print("[INFO] All required libraries are installed.")
        return True

# Import modules with safe fallback
try:
    from convert_edf_to_npy import main as preprocess_main
except ImportError as e:
    print("[ERROR] Could not import convert_edf_to_npy:", e)
    preprocess_main = None

try:
    from process_metadata_iteratively import main as metadata_main
except ImportError as e:
    print("[ERROR] Could not import process_metadata_iteratively:", e)
    metadata_main = None

try:
    from train_model_v2 import main as train_main
except ImportError as e:
    print("[ERROR] Could not import train_model_v2:", e)
    train_main = None

try:
    from evaluate_model import main as eval_main
except ImportError as e:
    print("[ERROR] Could not import evaluate_model:", e)
    eval_main = None

try:
    from explain_model import main as explain_main
except ImportError as e:
    print("[ERROR] Could not import explain_model:", e)
    explain_main = None


def run_all():
    print("[INFO] Starting the full pipeline...")

    if preprocess_main:
        print("[INFO] Running preprocessing (EDF to NPY)...")
        try:
            preprocess_main()
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return
        print("[INFO] Preprocessing completed.\n")
    
    if metadata_main:
        print("[INFO] Running metadata processing...")
        try:
            metadata_main()
        except Exception as e:
            print(f"[ERROR] Metadata processing failed: {e}")
            return
        print("[INFO] Metadata processing completed.\n")

    if train_main:
        print("[INFO] Running model training...")
        try:
            train_main()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return
        print("[INFO] Training completed.\n")

    if eval_main:
        print("[INFO] Running model evaluation...")
        try:
            eval_main()
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return
        print("[INFO] Evaluation completed.\n")

    if explain_main:
        print("[INFO] Running explanation/visualization...")
        try:
            explain_main()
        except Exception as e:
            print(f"[ERROR] Explanation failed: {e}")
            return
        print("[INFO] Explanation completed.\n")

    print("[INFO] All steps completed successfully!")


if __name__ == "__main__":
    if not check_requirements():
        sys.exit(1)
    run_all()
