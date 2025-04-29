###################
# Author - Gurleen Kaur
# Contributors - 
# File - load_data.py
# This file loads data from folders of both classes to be trained later
###################

import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
import random
from config import TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER
from sklearn.preprocessing import StandardScaler

# Define common EEG channels
COMMON_CHANNELS = [
    'EEG F7-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG T3-REF',
    'EEG P3-REF', 'EEG C4-REF', 'EEG PZ-REF', 'EEG T5-REF', 'EEG A1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'SUPPR', 'IBI', 'EEG P4-REF', 'EEG FZ-REF',
    'EEG T4-REF', 'EEG O1-REF', 'EEG F8-REF', 'BURSTS', 'EEG O2-REF',
    'EEG T6-REF', 'EEG F3-REF', 'EEG F4-REF'
]

# Fixed length for EEG data after processing
TIME_STEP = 120
TARGET_SAMPLING_RATE = 250 
FIXED_LENGTH =  TARGET_SAMPLING_RATE * TIME_STEP # Reduce this if memory is an issue
 # Reduced for better training efficiency

# Function to find the maximum EEG length in the dataset
def find_max_length(folder_paths):
    """Finds the maximum EEG signal length across all files."""
    max_length = FIXED_LENGTH  # Use a fixed value to prevent memory issues
    print(f"[INFO] Using fixed max length: {max_length} samples per EEG recording")
    return max_length

# Function to load and preprocess EEG data
def load_eeg(file_path, target_sampling_rate=TARGET_SAMPLING_RATE, fixed_length=FIXED_LENGTH):
    """Loads EEG data, selects common channels, resamples, and ensures fixed length."""
    # print(f"[INFO] Loading file: {file_path}")

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="error")
        original_sfreq = raw.info['sfreq']

        # Resampling for uniform sample rate
        if original_sfreq != target_sampling_rate:
            raw.resample(target_sampling_rate)
            # print(f"  - Resampled from {original_sfreq} Hz to {target_sampling_rate} Hz")

        available_channels = raw.ch_names
        selected_indices = [i for i, ch in enumerate(available_channels) if ch in COMMON_CHANNELS]

        if len(selected_indices) < len(COMMON_CHANNELS):
            print(f"[WARNING] {file_path} has only {len(selected_indices)} of {len(COMMON_CHANNELS)} common channels.")

        data = raw.get_data()[selected_indices, :]  # Shape: (channels, time)

        scaler = StandardScaler()
        data = scaler.fit_transform(data.T).T 
        # Debugging prints
        num_samples = data.shape[1]
        # print(f"  - EEG Data Shape Before Processing: {data.shape} (Channels, Time)")
        # print(f"  - Number of Samples Before Padding: {num_samples}")

        # Ensure fixed length
        if num_samples > fixed_length:
            data = data[:, :fixed_length]  # Truncate long signals
            # print(f"  - Truncated to: {fixed_length} samples")
        elif num_samples < fixed_length:
            pad_width = fixed_length - num_samples
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')  # Pad short signals
            # print(f"  - Padded to: {fixed_length} samples")

        # Final shape check
        # print(f"  - EEG Data Shape After Processing: {data.shape} (Channels, Time)")

        return data

    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

# Function to load a single EEG file using the existing load_eeg function
# Used later for model explaination
def load_single_file(eeg_file):
    """Loads a single EEG file for analysis instead of an entire dataset."""
    eeg_data = load_eeg(eeg_file)
    
    if eeg_data is None:
        raise ValueError(f"[ERROR] Failed to load EEG file: {eeg_file}")

    print(f"[INFO] Successfully loaded single EEG file: {eeg_file}")

    # Converting so that it can be compatible with CAM, SHAP fucntions
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)

    return eeg_data

def load_random_eeg_samples(folder_path, num_samples=5):
    """
    Loads a specified number of random EEG files from a folder.

    Parameters:
        folder_path (str): Path to the EEG folder containing .edf files.
        num_samples (int): Number of random EEG files to load.

    Returns:
        torch.Tensor: A batch of EEG tensors of shape (num_samples, channels, time).
    """
    # Get list of all .edf files in the folder
    eeg_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.edf')]
    
    if len(eeg_files) == 0:
        raise ValueError(f"[ERROR] No .edf files found in {folder_path}.")
    
    # Select random EEG files
    num_samples = min(num_samples, len(eeg_files))  # Ensure we don't select more than available
    selected_files = random.sample(eeg_files, num_samples)

    print(f"[INFO] Loading {num_samples} random EEG files from {folder_path}.")

    # Load EEG data from selected files
    eeg_data_list = []
    for file in selected_files:
        eeg_data = load_eeg(file)
        if eeg_data is not None:
            eeg_data_list.append(torch.tensor(eeg_data, dtype=torch.float32))

    # Stack EEG data into a single tensor batch (batch_size, channels, time)
    if len(eeg_data_list) == 0:
        raise ValueError("[ERROR] No valid EEG files were loaded.")

    eeg_batch = torch.stack(eeg_data_list)

    print(f"[INFO] Loaded batch shape: {eeg_batch.shape}")
    return eeg_batch

# EEG Dataset Class
class EEGDataset(Dataset):
    def __init__(self, abnormal_folder, normal_folder):
        self.file_paths = []
        self.labels = []

        for file in os.listdir(abnormal_folder):
            if file.endswith('.edf'):
                self.file_paths.append(os.path.join(abnormal_folder, file))
                self.labels.append(1)  # Abnormal Label

        for file in os.listdir(normal_folder):
            if file.endswith('.edf'):
                self.file_paths.append(os.path.join(normal_folder, file))
                self.labels.append(0)  # Normal Label

        # Find max length dynamically
        self.fixed_length = find_max_length([abnormal_folder, normal_folder])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        eeg_data = load_eeg(self.file_paths[idx], TARGET_SAMPLING_RATE, self.fixed_length)

        if eeg_data is None:
            print(f"[ERROR] Skipping file at index {idx} due to loading failure.")
            return None

        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    

if __name__ == "__main__":
    
    print("[INFO] Initializing EEG dataset...")
    dataset = EEGDataset(TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER)

    print(f"[INFO] Total EEG Files Loaded: {len(dataset)}")

    # Test first sample
    eeg_sample, label = dataset[0]
    print(f"[INFO] Sample EEG Shape: {eeg_sample.shape}, Label: {label}")