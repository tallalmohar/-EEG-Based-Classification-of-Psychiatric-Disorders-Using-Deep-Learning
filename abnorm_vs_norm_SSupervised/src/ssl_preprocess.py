###################
# Author - Michaela Foo 
# File - ssl_preprocess.py
# Description - 
# 1) Loads all .edf files from both normal and abnormal directories.
# 2) Extracts metadata (channels, frequency, subject ID, duration).
# 3) Preprocesses EEG signals (selects channels, resamples, normalizes, pads/truncates).
# 4) Saves preprocessed data and metadata.
###################

import os
import numpy as np
import mne
from glob import glob
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse

# === COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description="EEG Preprocessing Script")
parser.add_argument("--data_path", type=str, required=True, help="Path to the EEG data directory, ie ../train/train_data/")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed output, ie ../train/processed_train/")
args = parser.parse_args()

# === CONFIGURATION ===
DATA_PATH = args.data_path
NORMAL_DIR = os.path.join(DATA_PATH, "normal")
ABNORMAL_DIR = os.path.join(DATA_PATH, "abnormal")
OUTPUT_DIR = args.output_dir
TARGET_SAMPLING_RATE = 128  # Hz
TARGET_SIGNAL_DURATION = 234.375  # seconds (30,000 samples at 128 Hz)
TARGET_LENGTH = int(TARGET_SAMPLING_RATE * TARGET_SIGNAL_DURATION)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# === COMMON CHANNELS (predefined group decision) ===
COMMON_CHANNELS = [
    'EEG F7-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG T3-REF',
    'EEG P3-REF', 'EEG C4-REF', 'EEG PZ-REF', 'EEG T5-REF', 'EEG A1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'SUPPR', 'IBI', 'EEG P4-REF', 'EEG FZ-REF',
    'EEG T4-REF', 'EEG O1-REF', 'EEG F8-REF', 'BURSTS', 'EEG O2-REF',
    'EEG T6-REF', 'EEG F3-REF', 'EEG F4-REF'
]

# === UTILITY FUNCTIONS ===

def extract_subject_id(filename):
    return os.path.basename(filename).split("_")[0]

def load_eeg_file(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    return raw

def preprocess_eeg(raw, common_channels, target_sampling_rate, target_length):
    raw.pick_channels(common_channels)
    raw.resample(target_sampling_rate)
    eeg_data = raw.get_data()
    scaler = StandardScaler()
    eeg_data = scaler.fit_transform(eeg_data.T).T
    if eeg_data.shape[1] < target_length:
        pad_width = target_length - eeg_data.shape[1]
        eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant')
    elif eeg_data.shape[1] > target_length:
        eeg_data = eeg_data[:, :target_length]
    return eeg_data

def extract_metadata(raw, file_path, class_label):
    return {
        "file_name": os.path.basename(file_path),
        "subject_id": extract_subject_id(file_path),
        "class_label": class_label,
        "n_channels": raw.info['nchan'],
        "sampling_rate": raw.info['sfreq'],
        "duration_sec": raw.n_times / raw.info['sfreq'],
        "channel_names": raw.ch_names
    }

# === MAIN PROCESSING FUNCTION ===

def process_eeg_data(normal_dir, abnormal_dir, output_dir, target_sampling_rate, target_length):
    normal_files = glob(os.path.join(normal_dir, "*.edf"))
    abnormal_files = glob(os.path.join(abnormal_dir, "*.edf"))

    common_channels = COMMON_CHANNELS
    print(f"Total Normal Files: {len(normal_files)}")
    print(f"Total Abnormal Files: {len(abnormal_files)}")
    print(f"Common Channels: {common_channels}")
    print(f"Number of Common Channels: {len(common_channels)}")

    X_data, y_labels = [], []
    metadata = []
    channel_coverage = defaultdict(set)
    sampling_freqs = Counter()
    subject_ids = {"normal": set(), "abnormal": set()}

    print("\nProcessing normal EEG files...")
    for file in normal_files:
        try:
            raw = load_eeg_file(file)
            class_label = 0
            meta = extract_metadata(raw, file, class_label)
            metadata.append(meta)
            subject_ids["normal"].add(meta["subject_id"])
            for ch in meta["channel_names"]:
                channel_coverage[ch].add(meta["subject_id"])
            sampling_freqs[meta["sampling_rate"]] += 1
            signal = preprocess_eeg(raw, common_channels, target_sampling_rate, target_length)
            X_data.append(signal)
            y_labels.append(class_label)
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")

    print("\nProcessing abnormal EEG files...")
    for file in abnormal_files:
        try:
            raw = load_eeg_file(file)
            class_label = 1
            meta = extract_metadata(raw, file, class_label)
            metadata.append(meta)
            subject_ids["abnormal"].add(meta["subject_id"])
            for ch in meta["channel_names"]:
                channel_coverage[ch].add(meta["subject_id"])
            sampling_freqs[meta["sampling_rate"]] += 1
            signal = preprocess_eeg(raw, common_channels, target_sampling_rate, target_length)
            X_data.append(signal)
            y_labels.append(class_label)
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)
    np.save(os.path.join(output_dir, "X_eeg.npy"), X_data)
    np.save(os.path.join(output_dir, "y_labels.npy"), y_labels)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "eeg_metadata.csv"), index=False)

    print("\nEEG Data Preprocessing Complete!")
    print("\n==== Dataset Summary ====")
    print(f"Total Samples: {len(X_data)}")
    print(f"EEG Shape: {X_data.shape[1:]}")
    print(f"Common Channels Used: {len(common_channels)}")
    print(f"Original Sampling Frequencies: {dict(sampling_freqs)}")
    print(f"Subjects - Normal: {len(subject_ids['normal'])}, Abnormal: {len(subject_ids['abnormal'])}")
    print("Top Channels by Coverage:")
    for ch, subjects in sorted(channel_coverage.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {ch}: {len(subjects)} subjects")

    return metadata_df

# === ENTRY POINT ===

if __name__ == "__main__":
    process_eeg_data(NORMAL_DIR, ABNORMAL_DIR, OUTPUT_DIR, TARGET_SAMPLING_RATE, TARGET_LENGTH)
    print("\nEEG Data Preprocessing Completed.")
    print(f"Processed data saved to {OUTPUT_DIR}")
