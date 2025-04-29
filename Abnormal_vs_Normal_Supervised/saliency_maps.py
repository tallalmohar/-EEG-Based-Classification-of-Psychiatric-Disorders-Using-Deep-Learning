###################
# Author - Gurleen Kaur
# File - saliency_maps.py
###################

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import SALIENCY_MAP_PATH, FEATURE_IMPORTANCE_PATH

def compute_saliency_map(model, eeg_data, target_class):
    """Computes the saliency map for an EEG input per channel."""
    
    eeg_data = eeg_data.unsqueeze(0).requires_grad_(True)  # Add batch dimension
    model.eval()  # Set to evaluation mode

    output = model(eeg_data)
    model.zero_grad()

    # Compute gradient w.r.t. target class
    output[0, target_class].backward()
    saliency = eeg_data.grad.abs().squeeze().cpu().numpy()  # Absolute gradient values

    return saliency  # Shape: (num_channels, time_steps)


def visualize_saliency(model, eeg_data, target_class):
    """Runs saliency map computation and plots the result with all channels in different colors."""
    
    saliency_map = compute_saliency_map(model, eeg_data, target_class)
    num_channels, time_steps = saliency_map.shape

    target_class_name = "Normal" if target_class == 0 else "Abnormal"

    os.makedirs(os.path.dirname(SALIENCY_MAP_PATH), exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot each EEG channel with a different color
    for i in range(num_channels):
        plt.plot(saliency_map[i, :], label=f"Channel {i+1}")

    plt.title(f"Saliency Map for EEG Classification - {target_class_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Importance")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.grid()

    plt.savefig(SALIENCY_MAP_PATH, dpi=300)
    print(f"[INFO] Saliency map saved at {SALIENCY_MAP_PATH}")


def compute_top_features(model, eeg_data, target_class, channel_names):
    """Computes and returns the top 10 contributing EEG channels based on saliency map."""
    
    saliency_map = compute_saliency_map(model, eeg_data, target_class)

    # Compute importance per channel by averaging saliency values across all time steps
    channel_importance = np.mean(saliency_map, axis=1)  # Shape: (num_channels,)

    # Sort channels by importance and select top 10
    sorted_indices = np.argsort(channel_importance)[::-1]  # Descending order
    top_indices = sorted_indices[:10]
    top_importance = channel_importance[top_indices]

    # Get corresponding channel names
    top_channels = [channel_names[i] for i in top_indices]

    return top_channels, top_importance


def visualize_top_features(model, eeg_data, target_class, channel_names):
    """Plots and saves the top 10 EEG channels contributing to classification."""
    
    top_channels, top_importance = compute_top_features(model, eeg_data, target_class, channel_names)

    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.barh(top_channels[::-1], top_importance[::-1], color='red', alpha=0.7)  # Reverse order for best visualization
    plt.xlabel("Feature Importance Score")
    plt.ylabel("EEG Channels")
    plt.title("Top 10 Most Important EEG Channels (Saliency)")
    plt.grid()

    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300)
    print(f"[INFO] Feature importance bar chart saved at {FEATURE_IMPORTANCE_PATH}")
