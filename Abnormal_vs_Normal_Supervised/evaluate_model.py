###################
# Author - Gurleen Kaur
# Contributors - 
# File - evaluate_model.py
# This file evaluates the trained model on the test dataset and generates metrics.
###################

import torch
from define_model import EEGClassifier
from load_data_edf import EEGDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import TEST_ABNORMAL_FOLDER, TEST_NORMAL_FOLDER, MODEL_CHECKPOINT, CONFUSION_MATRIX_PATH, ROC_CURVE_PATH, TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER


# Function to Load Model Checkpoint
def load_model(device):
    """Loads the trained EEG classification model from a checkpoint."""
    
    model = EEGClassifier().to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    print("[INFO] Model loaded successfully!")
    
    return model


# Function to Load Test Dataset
def load_test_data(batch_size=32):
    """Loads test dataset and returns DataLoader."""
    
    dataset = EEGDataset(TEST_ABNORMAL_FOLDER, TEST_NORMAL_FOLDER)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Test Dataset Loaded: {len(dataset)} EEG files.")
    return dataset, dataloader


# Function to Evaluate Model
def evaluate_model(model, dataloader, device):
    """Evaluates the model and returns predictions, labels, and probability scores."""
    
    all_preds, all_labels = [], []
    y_true, y_scores = [], []

    with torch.no_grad():
        for eeg_data, labels in dataloader:
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            outputs = model(eeg_data)

            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    return all_preds, all_labels, y_true, y_scores


# Function to Plot & Save Confusion Matrix
def plot_confusion_matrix(cm, save_path=CONFUSION_MATRIX_PATH):
    """Plots and saves the confusion matrix."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Confusion matrix saved to {save_path}")


# Function to Plot & Save ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, save_path=ROC_CURVE_PATH):
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Random guess baseline

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] ROC curve saved to {save_path}")


# Main Execution
if __name__ == "__main__":
    # Device setup (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = load_model(device)

    # Load Test Data
    test_dataset, test_dataloader = load_test_data(batch_size=32)

    # Evaluate Model
    all_preds, all_labels, y_true, y_scores = evaluate_model(model, test_dataloader, device)

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Print Metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # Plot & Save Metrics
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)
