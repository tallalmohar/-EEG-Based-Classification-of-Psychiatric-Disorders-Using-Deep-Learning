# %% [markdown]
# # **Unsupervised Learning**
# ---

# %% [markdown]
# 
# Author - Liam Duncan \
# File - unsupervised-model.ipynb
# 

# %% [markdown]
# **Dataset:** 
# ---
# The dataset used in this study is obtained from the Temple University EEG\
# Corpus, which contains EEG recordings collected for psychiatric disorder clas sification. \
# The dataset includes both normal and abnormal EEG recordings,\
# categorized based on clinical evaluations.\
# Source: Lopez, S. (2017). Automated Identification of Abnormal EEGs. 
#  Temple University. \
#  https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

# %% [markdown]
# **Decisions:**
# ---
# All values chosen throughout this notebook were chosen based on a smaller subset of the dataset\
# 500 files were explored ~250 normal and ~250 abnormal. Experiments regarding clustering algorithms,\
# the number of principle components, z-score thresholds, cluster initialization techniques, etc.\
# Based on the results found in the data exploration, the optimal model was chosen
# 
# **Model**
# ---
# Clustering algorithm: Kmeans\
# Number of Clusters: 2\
# Cluster Initilization: K-means++\
# N_init: 1000\
# Number of Principle Components: 2\
# Z-score Threshold: >22 seen as outliers\
# Scaling technique: Standardization

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Import Modules

# %%
from collections import Counter
import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import stats
from scipy.stats import mode
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ### Loading Data
# **Update file paths**
# 
# Code altered from load_data.py

# %%

# Define common EEG channels
COMMON_CHANNELS = [
    'EEG F7-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG T3-REF',
    'EEG P3-REF', 'EEG C4-REF', 'EEG PZ-REF', 'EEG T5-REF', 'EEG A1-REF',
    'EEG FP2-REF', 'EEG FP1-REF', 'SUPPR', 'IBI', 'EEG P4-REF', 'EEG FZ-REF',
    'EEG T4-REF', 'EEG O1-REF', 'EEG F8-REF', 'BURSTS', 'EEG O2-REF',
    'EEG T6-REF', 'EEG F3-REF', 'EEG F4-REF'
]

# Fixed length for EEG data after processing
FIXED_LENGTH = 30000  # Reduce this if memory is an issue
TARGET_SAMPLING_RATE = 250  # Reduced for better training efficiency

# Function to find the maximum EEG length in the dataset
def find_max_length(folder_paths):
    """Finds the maximum EEG signal length across all files."""
    max_length = FIXED_LENGTH  # Use a fixed value to prevent memory issues
    print(f"[INFO] Using fixed max length: {max_length} samples per EEG recording")
    return max_length

# Function to load and preprocess EEG data
def load_eeg(file_path, target_sampling_rate=TARGET_SAMPLING_RATE, fixed_length=FIXED_LENGTH):
    """Loads EEG data, selects common channels, resamples, and ensures fixed length."""
    print(f"[INFO] Loading file: {file_path}")

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        original_sfreq = raw.info['sfreq']

        # Resampling for uniform sample rate
        if original_sfreq != target_sampling_rate:
            raw.resample(target_sampling_rate)
            print(f"  - Resampled from {original_sfreq} Hz to {target_sampling_rate} Hz")

        available_channels = raw.ch_names
        selected_indices = [i for i, ch in enumerate(available_channels) if ch in COMMON_CHANNELS]

        if len(selected_indices) < len(COMMON_CHANNELS):
            print(f"[WARNING] {file_path} has only {len(selected_indices)} of {len(COMMON_CHANNELS)} common channels.")

        data = raw.get_data()[selected_indices, :]  # Shape: (channels, time)

        # Debugging prints
        num_samples = data.shape[1]
        print(f"  - EEG Data Shape Before Processing: {data.shape} (Channels, Time)")
        print(f"  - Number of Samples Before Padding: {num_samples}")

        # Ensure fixed length
        if num_samples > fixed_length:
            data = data[:, :fixed_length]  # Truncate long signals
            print(f"  - Truncated to: {fixed_length} samples")
        elif num_samples < fixed_length:
            pad_width = fixed_length - num_samples
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')  # Pad short signals
            print(f"  - Padded to: {fixed_length} samples")

        # Final shape check
        print(f"  - EEG Data Shape After Processing: {data.shape} (Channels, Time)")

        return data

    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

# EEG Dataset Class for Unsupervised Learning (No Labels)
# Altered to not use tensor as not needed for unsupervised learning
class EEGDataset:
    def __init__(self, abnormal_folder, normal_folder):
        self.file_paths = []

        for file in os.listdir(abnormal_folder):
            if file.endswith('.edf'):
                self.file_paths.append(os.path.join(abnormal_folder, file))

        for file in os.listdir(normal_folder):
            if file.endswith('.edf'):
                self.file_paths.append(os.path.join(normal_folder, file))

        self.fixed_length = find_max_length([abnormal_folder, normal_folder])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        eeg_data = load_eeg(self.file_paths[idx], TARGET_SAMPLING_RATE, self.fixed_length)

        if eeg_data is None:
            print(f"[ERROR] Skipping file at index {idx} due to loading failure.")
            return None

        return eeg_data

# %% [markdown]
############# Training Data ##########

# %%
# UPDATE PATHS 
train_abnormal_folder = r"../abnormal_train_data"
train_normal_folder = r"../normal_train_data"

print("[INFO] Initializing EEG dataset...")
all_train_dataset = EEGDataset(train_abnormal_folder, train_normal_folder)

print(f"[INFO] Total EEG Files Loaded: {len(all_train_dataset)}")

# Extract EEG data
all_train_data = []

for idx in range(len(all_train_dataset)):
    eeg_data = all_train_dataset[idx]
    if eeg_data is not None:
        all_train_data.append(eeg_data.flatten())
        # Each sample is flattened to allow for clustering as easier with 2D data

all_train_data = np.array(all_train_data)

# %%
# This  section is used for getting ground truth labels and for plotting the abnormal data on its own
train_abnormal_data = []
train_normal_data = []

for idx, file_path in enumerate(all_train_dataset.file_paths):
    if train_abnormal_folder in file_path:
        train_abnormal_data.append(all_train_data[idx])
    else:
        train_normal_data.append(all_train_data[idx])

train_abnormal_data = np.array(train_abnormal_data)
train_normal_data = np.array(train_normal_data)

train_abnormal_length = (len(train_abnormal_data))
train_normal_length = (len(train_normal_data))


# %% [markdown]
############# TEST Data ##########

# %%
# UPDATE PATHS
test_abnormal_folder = r"../abnormal_test_data"
test_normal_folder = r"../normal_test_data"

print("[INFO] Initializing EEG dataset...")
test_dataset = EEGDataset(test_abnormal_folder, test_normal_folder)

print(f"[INFO] Total EEG Files Loaded: {len(test_dataset)}")

# Extract EEG data
test_data = []

for idx in range(len(test_dataset)):
    eeg_data = test_dataset[idx]
    if eeg_data is not None:
        test_data.append(eeg_data.flatten())
        # Each sample is flattened to allow for clustering as easier with 2D data

test_data = np.array(test_data)

# %%
# This  section is used for splitting data so ground truth labels of test data can be retrieved
test_abnormal_data = []
test_normal_data = []

for idx, file_path in enumerate(test_dataset.file_paths):
    if test_abnormal_folder in file_path:
        test_abnormal_data.append(test_data[idx])
    else:
        test_normal_data.append(test_data[idx])

test_abnormal_data = np.array(test_abnormal_data)
test_normal_data = np.array(test_normal_data)

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Data Preprocessing - Outlier Removal, Standard Scaling, PCA

# %% [markdown]
# ##### Train Data

# %%
# Standardize the data for better clustering performance
print("[INFO] Scaling Training Dataset")
scaler = StandardScaler()
all_train_data_sclaed = scaler.fit_transform(all_train_data)

# %%
# Labels (0 = Abnormal, 1 = Normal)
labels = np.array([0] * train_abnormal_length + [1] * train_normal_length)
colors = np.array(["red" if label == 0 else "blue" for label in labels])
label_names = {0: "Abnormal", 1: "Normal"}


# t-SNE Visualization 
print("[INFO] Performing t-SNE Visualization")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
X_tsne = tsne.fit_transform(all_train_data_sclaed)

plt.figure(figsize=(10, 5))
for label, color in zip([0, 1], ["red", "blue"]):
    plt.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1], 
                c=color, label=label_names[label], alpha=0.7)
plt.title("t-SNE Visualization of EEG Training Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="EEG Type")
plt.savefig("t-SNE_visualization.png", dpi=300, bbox_inches="tight")
print("[INFO] Plot saved as: t-SNE_visualization.png")
plt.close()


# %%
print("[INFO] Performing KMeans Clustering on Abnormal Data")
abnormal_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=100)
final_labels = abnormal_kmeans.fit_predict(train_abnormal_data)
pca = PCA(n_components=2)
train_abnormal_data_pca = pca.fit_transform(train_abnormal_data)
sns.scatterplot(x=train_abnormal_data_pca[:, 0], y=train_abnormal_data_pca[:, 1], hue=final_labels, palette='viridis', legend='full')
plt.title("K-Means Clustering on Abnormal Training Data")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("abnormal_kmeans_plot.png", dpi=300, bbox_inches="tight")
print("[INFO] Plot saved as: abnormal_kmeans_plot.png")
plt.close()


# %%
print("[INFO] Removing Outliers and Performing PCA on Training Data")
# Compute Z-scores
z_scores = np.abs(stats.zscore(all_train_data_sclaed))

# Remove outliers based on the current threshold
train_data_no_outliers = all_train_data_sclaed[(z_scores < 22).all(axis=1)]

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_no_outliers)

# %% [markdown]
# ##### Test Data
# Do not remove outliers of Test Data

# %%
print("------------------------------------------------")
print("[INFO] Performing Scaling and PCA on Test Data")
# Standardize the data for better clustering performance
test_data_scaled = scaler.fit_transform(test_data)

# Reduce dimensionality for visualization
test_data_pca = pca.fit_transform(test_data_scaled)

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### KMeans Clustering

# %%
print("[INFO] Performing KMeans Clustering on Training Data")
kmeans = KMeans(n_clusters=2, random_state=47, n_init=100, init='k-means++')
kmeans_labels = kmeans.fit_predict(train_data_no_outliers)

# %% [markdown]
# #### Cluster Plotting Function

# %%
# Function to plot clustering results

def plot_clusters(data, labels, title):
    # Create a safe filename by replacing spaces and special characters
    filename = re.sub(r'[^\w\-_]', '_', title) + ".png"
    
    # Plot the clusters
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis', legend='full')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Save the figure instead of showing it
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot saved as: {filename}")

    # Close the figure to free memory
    plt.close()

# %% [markdown]
# #### Model Analysis - Training

# %%
# Generate ground truth labels
ground_truth_labels = np.zeros(len(train_data_pca))  
ground_truth_labels[train_abnormal_length:] = 1  

# KMeans assigns arbitrary labels (0/1), so we need to map them to the ground truth
mapped_labels = np.zeros_like(kmeans_labels)

# Find the most common ground truth label for each cluster and assign mapped labels
for cluster in np.unique(kmeans_labels):
    mask = kmeans_labels == cluster
    most_common_label = mode(ground_truth_labels[mask], keepdims=True).mode[0]
    mapped_labels[mask] = most_common_label

# Compute accuracy
correct_assignments = np.sum(mapped_labels == ground_truth_labels)
total_points = len(ground_truth_labels)
accuracy = (correct_assignments / total_points) * 100
print("------------------------------------------------")
print("[Results] Training Data")
print("------------------------------------------------")
print(f"Correctly clustered points: {accuracy}%")
train_sil_score = silhouette_score(train_data_pca, kmeans_labels)
train_dbi_score = davies_bouldin_score(train_data_pca, kmeans_labels)
train_ari_score = adjusted_rand_score(ground_truth_labels, kmeans_labels)
print(f"Silhouette Score for Train Data: {train_sil_score} ")
print(f"Davies-Bouldin Index Score for Train Data: {train_dbi_score} ")
print(f"Adjusted Rand Index Score for Train Data: {train_ari_score} ")
print("Abnormal Data Label: 0")
print("Normal Data Label: 1")
plot_clusters(train_data_pca, kmeans_labels, "KMeans Clustering of All Train Data")
print("------------------------------------------------")

# %% [markdown]
# ---
# ---

# %% [markdown]
# ### Model Prediction and Analysis - Test Data

# %%
print("[INFO] Performing KMeans Clustering on Test Data")
test_cluster_labels = kmeans.predict(test_data_scaled)

# %%
abnormal_test_size = len(test_abnormal_data)

abnormal_segment = test_cluster_labels[:abnormal_test_size]  # Abnormal data
normal_segment = test_cluster_labels[abnormal_test_size:]   # Normal data

# Find the majority cluster in each segment
majority_cluster_1 = Counter(abnormal_segment).most_common(1)[0][0]  # Abnormal data
majority_cluster_2 = Counter(normal_segment).most_common(1)[0][0]  # Normal data

print(f"Majority cluster for abnormal data: {majority_cluster_1}")
print(f"Majority cluster for normal data: {majority_cluster_2}")

# Update cluster-to-label mapping based on majority cluster identification
cluster_to_label = {
    majority_cluster_1: 0,  # Abnormal: 0
    majority_cluster_2: 1   # Normal: 1
}

# Assign labels to test data
test_pred_labels = np.array([cluster_to_label.get(c, 0) for c in test_cluster_labels])

# Ground truth (abnormal: 0, normal: 1)
y_test = np.array([0] * abnormal_test_size + [1] * (len(test_cluster_labels) - abnormal_test_size))

# Compute accuracy
print("------------------------------------------------")
print("[Results] Testing Data")
print("------------------------------------------------")
accuracy = accuracy_score(y_test, test_pred_labels)
print(f"Test Clustering Accuracy: {accuracy}")
test_sil_score = silhouette_score(test_data_pca, test_cluster_labels)
test_dbi_score = davies_bouldin_score(test_data_pca, test_cluster_labels)
test_ari_score = adjusted_rand_score(y_test, test_cluster_labels)
print(f"Silhouette Score for Test Data: {test_sil_score}")
print(f"Davies-Bouldin Index Score for Test Data: {test_dbi_score} ")
print(f"Adjusted Rand Index Score for Test Data: {test_ari_score} ")
plot_clusters(test_data_pca, test_cluster_labels, "KMeans Clustering of Test Data")
print("------------------------------------------------")