###################
# Author - Gurleen Kaur
# Contributors - 
# File - convert_edf_to_npy.py
###################

import os
import numpy as np
from load_data_edf import load_eeg  # make sure this has COMMON_CHANNELS, FIXED_LENGTH defined
from config import TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER

SAVE_ROOT = "npy_data"
SAVE_ABNORMAL = os.path.join(SAVE_ROOT, "abnormal")
SAVE_NORMAL = os.path.join(SAVE_ROOT, "normal")

os.makedirs(SAVE_ABNORMAL, exist_ok=True)
os.makedirs(SAVE_NORMAL, exist_ok=True)

def convert_and_save(folder_path, save_path, label):
    for file in os.listdir(folder_path):
        if file.endswith(".edf"):
            file_path = os.path.join(folder_path, file)
            eeg_data = load_eeg(file_path)
            if eeg_data is not None:
                save_file = os.path.join(save_path, file.replace(".edf", ".npy"))
                np.save(save_file, eeg_data)
                print(f"[INFO] Saved {label} EEG to: {save_file}")

if __name__ == "__main__":
    print("[INFO] Converting abnormal EEG files...")
    convert_and_save(TRAIN_ABNORMAL_FOLDER, SAVE_ABNORMAL, label="abnormal")

    print("[INFO] Converting normal EEG files...")
    convert_and_save(TRAIN_NORMAL_FOLDER, SAVE_NORMAL, label="normal")

    print("[DONE] All files converted and saved as .npy.")
