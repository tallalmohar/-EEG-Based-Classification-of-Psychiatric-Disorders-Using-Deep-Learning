import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, TensorDataset
from config import NPY_DATA_DIR, MODEL_CHECKPOINT, LOSS_CURVE_PATH, ACCURACY_CURVE_PATH
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler

# ------------------------
class EEGDataset(Dataset):
    def __init__(self, root_dir, labeled=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        self.data = []

        for label, folder in enumerate(['normal', 'abnormal']):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                self.data.append((os.path.join(folder_path, file), label))

        if not labeled:
            self.data = [(path, None) for path, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        eeg = np.load(path).astype(np.float32)
        eeg = eeg[:, :2048]
        eeg = torch.tensor(eeg).unsqueeze(0)

        if self.transform:
            eeg = self.transform(eeg)

        if self.labeled:
            return eeg, torch.tensor(label, dtype=torch.long)
        else:
            return eeg

# ------------------------


class EEGClassifier(nn.Module):
    def __init__(self, num_channels=24, num_classes=2, mlp_hidden_size=128):
        super(EEGClassifier, self).__init__()

        # CNN Feature Extraction
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)  # Adjusted input channels to 1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        # Fully Connected (MLP) Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * (num_channels // 2) * (num_channels // 2), mlp_hidden_size)
        self.fc2 = nn.Linear(mlp_hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch, 1, channels, time)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)  # Flatten for MLP
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Final classification layer
        return x



# ------------------------
def get_balanced_loader(dataset, batch_size):
    labels = [label for _, label in dataset]
    class_sample_count = np.bincount(labels)
    weight = 1. / class_sample_count
    samples_weight = [weight[label] for label in labels]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ------------------------
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# ------------------------
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            preds = (probs[:, 1] > threshold).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return avg_loss, accuracy

# ------------------------
def find_best_threshold(model, dataloader, device):
    model.eval()
    best_threshold = 0.5
    best_f1 = 0
    thresholds = np.arange(0.3, 0.9, 0.05)
    with torch.no_grad():
        for t in thresholds:
            all_preds, all_labels = [], []
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                probs = torch.softmax(model(x), dim=1)
                preds = (probs[:, 1] > t).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
    print(f"[INFO] Optimal threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}")
    return best_threshold

# ------------------------
def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(LOSS_CURVE_PATH)

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy", color='green')
    plt.plot(val_accuracies, label="Validation Accuracy", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.savefig(ACCURACY_CURVE_PATH)

# ------------------------
def generate_pseudo_labels(model, dataloader, device, threshold=0.9):
    model.eval()
    pseudo_data, pseudo_labels = [], []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            for i in range(len(confidence)):
                if confidence[i] > threshold:
                    pseudo_data.append(x[i].cpu())
                    pseudo_labels.append(pred[i].cpu())
    return pseudo_data, pseudo_labels

# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    labeled_dataset = EEGDataset(NPY_DATA_DIR, labeled=True)
    labeled_size = int(0.4 * len(labeled_dataset))
    val_size = int(0.1 * len(labeled_dataset))
    train_labeled, val_dataset, _ = random_split(labeled_dataset, [labeled_size, val_size, len(labeled_dataset) - labeled_size - val_size])
    unlabeled_dataset = EEGDataset(NPY_DATA_DIR, labeled=False)

    labels_np = np.array([label for _, label in train_labeled])
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    train_loader = get_balanced_loader(train_labeled, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8)

    model = EEGClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT)
            print(f" New best model saved (val_loss={val_loss:.4f}) to {MODEL_CHECKPOINT}")

    threshold = find_best_threshold(model, val_loader, device)
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

    print("Generating pseudo-labels...")
    pseudo_data, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, device, threshold=0.9)
    print(f"Added {len(pseudo_data)} pseudo-labeled examples.")

    NUM_PSEUDO_ROUNDS = 2
    combined_dataset = train_labeled

    for round in range(NUM_PSEUDO_ROUNDS):
        print(f"Pseudo-labeling Round {round + 1}")
        pseudo_threshold = max(0.9 - round * 0.1, 0.6)
        combined_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
        for epoch in range(15):
            train_loss, train_acc = train(model, combined_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, threshold=threshold)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch + 1} - Semi-supervised Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_CHECKPOINT)
                print(f"New best model saved (val_loss={val_loss:.4f}) to {MODEL_CHECKPOINT}")

        pseudo_data, pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, device, threshold=pseudo_threshold)
        print(f"Added {len(pseudo_data)} pseudo-labeled examples.")
        if pseudo_data:
            pseudo_dataset = TensorDataset(torch.stack(pseudo_data), torch.stack(pseudo_labels))
            combined_dataset = ConcatDataset([combined_dataset, pseudo_dataset])

    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)
