###################
# Author - Gurleen Kaur
# File - train_model.py
###################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from define_model import EEGClassifier
# from load_data_edf import EEGDataset
from load_data_npy import NPYEEGDataset 
from sklearn.metrics import accuracy_score
from config import (
    TRAIN_ABNORMAL_FOLDER, TRAIN_NORMAL_FOLDER, MODEL_CHECKPOINT, LOSS_CURVE_PATH, 
    MODEL_DIR, ACCURACY_CURVE_PATH, NPY_ABNORMAL_FOLDER, NPY_NORMAL_FOLDER, NUM_EPOCHS
)

from torch.utils.tensorboard import SummaryWriter


# Function to plot & save loss and accuracy curves
def plot_loss_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history,
                       loss_path=LOSS_CURVE_PATH, acc_path=ACCURACY_CURVE_PATH):
    """Plots and saves the training/validation loss and accuracy curves."""

    os.makedirs(os.path.dirname(loss_path), exist_ok=True)
    
    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Training Loss", color='blue', linewidth=2)
    plt.plot(val_loss_history, label="Validation Loss", color='red', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(loss_path, dpi=300)
    print(f"[INFO] Loss curve plot saved to {loss_path}")
    
    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label="Training Accuracy", color='blue', linewidth=2)
    plt.plot(val_acc_history, label="Validation Accuracy", color='green', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.savefig(acc_path, dpi=300)
    print(f"[INFO] Accuracy curve plot saved to {acc_path}")


# Function to initialize dataset, split into train/validation, and return dataloaders
def load_data(batch_size=4, val_split=0.2):
    """Loads EEG dataset, splits into train/validation, and returns DataLoaders."""
    
    dataset = NPYEEGDataset(NPY_ABNORMAL_FOLDER, NPY_NORMAL_FOLDER)

    if len(dataset) == 0:
        raise ValueError("[ERROR] No EEG data found. Check dataset paths in config.py!")

    # Splitting dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {len(dataset)} EEG files")
    print(f"  - Training Set: {train_size} samples")
    print(f"  - Validation Set: {val_size} samples")

    return train_loader, val_loader


# Function to initialize model, loss function, and optimizer
def initialize_model(device):
    """Initializes model and optimizer, loads checkpoint if available."""
    
    model = EEGClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    start_epoch = 0

    # Uncomment the checkpoint code if you wish to load an existing model.
    # if os.path.exists(MODEL_CHECKPOINT):
    #     checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"[INFO] Loaded checkpoint from epoch {start_epoch}")
    # else:
    #     print("[INFO] No checkpoint found, starting training from scratch.")

    return model, criterion, optimizer, start_epoch


# Function to evaluate model on validation set: returns loss and accuracy.
def evaluate_validation(model, val_loader, criterion, device):
    """Evaluates model on validation data and returns loss and accuracy."""
    
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for eeg_data, labels in tqdm(val_loader, desc="Evaluating"):
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100  # Convert to percentage
    return avg_loss, accuracy


# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, val_interval=1, log_dir="runs/eeg_experiment"):
    """Trains the model, evaluates validation loss and accuracy, and saves checkpoints."""

    writer = SummaryWriter(log_dir=log_dir)
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for eeg_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds) * 100

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)

        # Validation evaluation at the specified interval
        if (epoch + 1) % val_interval == 0:
            val_loss, val_accuracy = evaluate_validation(model, val_loader, criterion, device)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)
            print(f"[INFO] Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            writer.add_scalar("Loss/validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch + 1)
        else:
            # If not evaluating, record placeholders (or you can choose to repeat the last value)
            val_loss_history.append(None)
            val_acc_history.append(None)

        # Save model checkpoint each epoch
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, MODEL_CHECKPOINT)
        print(f"[INFO] Model checkpoint saved at epoch {epoch+1}")

    writer.close()
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


# Function to load the trained model
def load_trained_model(model, device):
    """Loads trained model for evaluation."""
    
    if os.path.exists(MODEL_CHECKPOINT):
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("[INFO] Model loaded successfully!")
    else:
        print("[WARNING] No trained model found for evaluation!")


# Main Execution
if __name__ == "__main__":
    # Setup device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data (Train/Validation Split)
    train_loader, val_loader = load_data(batch_size=8)

    # Initialize Model
    model, criterion, optimizer, start_epoch = initialize_model(device)

    # Train Model
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device=device
    )

    # Load Trained Model for Evaluation
    load_trained_model(model, device)

    # Remove placeholder values (if any) from validation histories before plotting
    # This step is optional, depending on whether you want a continuous curve or only evaluated epochs.
    eval_epochs = [i+1 for i, v in enumerate(val_loss_history) if v is not None]
    eval_val_loss = [v for v in val_loss_history if v is not None]
    eval_val_acc = [v for v in val_acc_history if v is not None]

    # Plot Loss & Accuracy Curves (using only epochs with validation evaluations)
    plot_loss_accuracy(train_loss_history, eval_val_loss, train_acc_history, eval_val_acc)
