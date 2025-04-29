import os
import mne
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from config import TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER, TEST_ABNORMAL_FOLDER, TEST_NORMAL_FOLDER

##############################
# Metadata Extraction        #
##############################

def get_edf_files(folder):
    """Returns a list of full paths for all EDF files in the given folder."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.edf')]

def extract_subject_id(filename):
    """
    Extracts the subject ID from the filename.
    For example, for "aaaaaaaq_s004_t000.edf", returns "aaaaaaaq".
    """
    return os.path.basename(filename).split("_")[0]

def read_edf_metadata(reader, filename):
    """
    Extracts metadata from an open EDF file.
    Uses current MNE attributes.
    
    Returns a dictionary with:
      - file_name: The name of the EDF file.
      - n_channels: Number of channels.
      - signal_labels: List of channel names.
      - sampling_rates: List of sampling frequencies (assumes same rate for all channels).
      - samples_per_channel: Number of samples per channel.
      - start_time: Measurement date (if available).
      - file_duration: Duration of the recording in seconds.
    """
    # Get channel names from reader
    signal_labels = reader.ch_names  
    n_signals = len(signal_labels)
    
    # Get sampling rate from info (assumes same for all channels)
    sampling_rate = reader.info.get("sfreq", None)
    sampling_rates = [sampling_rate] * n_signals if sampling_rate is not None else [None] * n_signals
    
    # Number of samples for each channel (usually the same for all channels)
    nsamples = [reader.n_times] * n_signals
    
    # Get measurement date if available
    start_time = reader.info.get("meas_date", None)
    
    # Compute file duration (n_times divided by sampling rate)
    file_duration = reader.n_times / sampling_rate if sampling_rate else None

    metadata = {
        "file_name": filename,
        "n_channels": n_signals,
        "signal_labels": signal_labels,
        "sampling_rates": sampling_rates,
        "samples_per_channel": nsamples,
        "start_time": start_time,
        "file_duration": file_duration
    }
    return metadata

def process_metadata_for_folder(folder, class_label):
    """
    Processes all EDF files in a given folder and extracts metadata.
    Adds 'class_label' and 'Person_ID' fields.
    """
    metadata_list = []
    if not os.path.exists(folder):
        print(f"[Error] Folder '{folder}' does not exist.")
        return metadata_list

    edf_files = get_edf_files(folder)
    if not edf_files:
        print(f"[Warning] No EDF files found in '{folder}'.")
        return metadata_list

    for file_path in edf_files:
        try:
            reader = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            meta = read_edf_metadata(reader, os.path.basename(file_path))
            meta["class_label"] = class_label
            meta["Person_ID"] = extract_subject_id(meta["file_name"])
            metadata_list.append(meta)
            reader.close()
        except Exception as e:
            print(f"[Warning] Failed to process file '{file_path}': {e}")
    return metadata_list

def process_all_metadata(folders_with_labels):
    """
    Processes metadata for multiple folders.
    folders_with_labels: list of tuples (folder_path, class_label)
    Returns a combined list of metadata dictionaries.
    """
    all_metadata = []
    for folder, label in folders_with_labels:
        meta = process_metadata_for_folder(folder, label)
        all_metadata.extend(meta)
    return all_metadata

##############################
# Aggregation & Analysis     #
##############################

def aggregate_metadata(metadata_list):
    """
    Aggregates metadata from a list of metadata dictionaries.
    
    Returns:
      - unique_subjects: set of all subject IDs.
      - channel_subject_counts: dict mapping each channel to count of unique subjects having that channel.
      - freq_counter: Counter for sampling frequencies (one count per file).
    """
    channel_to_subjects = defaultdict(set)
    freq_counter = Counter()
    unique_subjects = set()

    for meta in metadata_list:
        subj = meta["Person_ID"]
        unique_subjects.add(subj)
        for ch in meta["signal_labels"]:
            channel_to_subjects[ch].add(subj)
        if meta["sampling_rates"]:
            # Assume all channels have same sampling frequency; count first one.
            freq = meta["sampling_rates"][0]
            freq_counter[freq] += 1

    channel_subject_counts = {ch: len(subjs) for ch, subjs in channel_to_subjects.items()}
    return unique_subjects, channel_subject_counts, freq_counter

def find_common_channels(metadata_list):
    """
    Finds the set of channels that are common across all files in the metadata list.
    Returns a set of channels.
    """
    if not metadata_list:
        return set()
    channel_sets = [set(meta["signal_labels"]) for meta in metadata_list]
    common_channels = set.intersection(*channel_sets)
    return common_channels

def display_metadata_summary(total_files, unique_subjects, channel_subject_counts, log_list=None, title="EEG Metadata Summary"):
    """
    Displays and optionally stores a summary of the metadata.
    If log_list is provided, it appends summary strings to it.
    """
    common_channels = [ch for ch, count in channel_subject_counts.items() if count == len(unique_subjects)]
    
    lines = [
        f"==== {title} ====",
        f"Total EDF Files: {total_files}",
        f"Total Unique Subjects: {len(unique_subjects)}",
        f"Total Common Channels (present in all subjects): {len(common_channels)}",
    ]

    print("\n".join(lines))
    
    if log_list is not None:
        log_list.extend(lines)
        log_list.append("") 


##############################
# Plotting Functions         #
##############################

def plot_histogram(data_dict, xlabel, ylabel, title, width=0.8):
    """Plots a bar chart from a dictionary."""
    keys = list(data_dict.keys())
    counts = list(data_dict.values())
    plt.figure(figsize=(12, 6))
    plt.bar(keys, counts, width=width, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_channel_histogram_subjects(metadata_list, class_label):
    """
    Plots a histogram for a given class label showing each EEG channel versus the number of unique subjects.
    """
    subset = [meta for meta in metadata_list if meta.get("class_label") == class_label]
    channel_to_subjects = defaultdict(set)
    for meta in subset:
        subj = meta["Person_ID"]
        for ch in meta["signal_labels"]:
            channel_to_subjects[ch].add(subj)
    channel_subject_counts = {ch: len(subjs) for ch, subjs in channel_to_subjects.items()}
    plot_histogram(channel_subject_counts, "EEG Channel", "Number of Unique Subjects",
                   f"Histogram of Unique Subjects per EEG Channel ({class_label.capitalize()})")

def plot_sampling_frequency_histogram(metadata_list, class_label):
    """
    Plots a histogram for a given class label showing sampling frequencies versus the number of files.
    """
    subset = [meta for meta in metadata_list if meta.get("class_label") == class_label]
    freq_counter = Counter()
    for meta in subset:
        if meta["sampling_rates"]:
            freq = meta["sampling_rates"][0]
            freq_counter[freq] += 1
    plot_histogram(freq_counter, "Sampling Frequency (Hz)", "Number of Files",
                   f"Histogram of Files per Sampling Frequency ({class_label.capitalize()})", width=1.5)

##############################
# Main Function              #
##############################

def main():
    summary_log = []  # List to collect summary lines for saving

    abnormal_folder = TEST_ABNORMAL_FOLDER
    normal_folder = TEST_NORMAL_FOLDER
    folders_with_labels = [(abnormal_folder, "abnormal"), (normal_folder, "normal")]

    all_metadata = process_all_metadata(folders_with_labels)
    if not all_metadata:
        print("[Error] No metadata extracted. Check your folder paths and EDF files.")
        return

    total_files = len(all_metadata)
    unique_subjects, channel_subject_counts, freq_counter = aggregate_metadata(all_metadata)

    # Log and print overall summary
    display_metadata_summary(total_files, unique_subjects, channel_subject_counts, log_list=summary_log, title="Overall EEG Metadata Summary")

    # Plot overall histogram
    plot_histogram(freq_counter, "Sampling Frequency (Hz)", "Number of Files",
                   "Histogram of Files per Sampling Frequency (Overall)", width=1.5)

    for class_label in ["abnormal", "normal"]:
        class_metadata = [meta for meta in all_metadata if meta.get("class_label") == class_label]
        if not class_metadata:
            warning = f"[Warning] No metadata for class {class_label}."
            print(warning)
            summary_log.append(warning)
            continue

        total_files_class = len(class_metadata)
        unique_subjects_class, channel_subject_counts_class, freq_counter_class = aggregate_metadata(class_metadata)

        # Log and print class summary
        display_metadata_summary(
            total_files_class,
            unique_subjects_class,
            channel_subject_counts_class,
            log_list=summary_log,
            title=f"{class_label.capitalize()} EEG Metadata Summary"
        )

        # Plot histograms
        plot_channel_histogram_subjects(all_metadata, class_label)
        plot_sampling_frequency_histogram(all_metadata, class_label)

    # Save all summary text
    with open(r"D:\SFU\CMPT340\Project\2025_1_project_10\edf_metadata_summary.txt", "w") as f:
        for line in summary_log:
            f.write(line + "\n")
    print("[INFO] Summary saved to 'edf_metadata_summary.txt'")


if __name__ == "__main__":
    main()
