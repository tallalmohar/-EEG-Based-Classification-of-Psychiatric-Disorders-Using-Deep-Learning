###################
# Author - Gurleen Kaur
# File - load_npy_data.py
# Description - Loads EEG data from preprocessed .npy files
###################

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import NPY_ABNORMAL_FOLDER, NPY_NORMAL_FOLDER

class NPYEEGDataset(Dataset):
    def __init__(self, abnormal_folder, normal_folder):
        self.file_paths = []
        self.labels = []

        for file in os.listdir(abnormal_folder):
            if file.endswith('.npy'):
                self.file_paths.append(os.path.join(abnormal_folder, file))
                self.labels.append(1)  # Abnormal label

        for file in os.listdir(normal_folder):
            if file.endswith('.npy'):
                self.file_paths.append(os.path.join(normal_folder, file))
                self.labels.append(0)  # Normal label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        data = np.load(file_path)
        eeg_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return eeg_tensor, label_tensor

if __name__ == "__main__":
    print("[INFO] Loading EEG .npy dataset...")
    dataset = NPYEEGDataset(NPY_ABNORMAL_FOLDER, NPY_NORMAL_FOLDER)

    print(f"[INFO] Total NPY EEG Files Loaded: {len(dataset)}")
    eeg_sample, label = dataset[0]
    print(f"[INFO] Sample EEG Shape: {eeg_sample.shape}, Label: {label}")
