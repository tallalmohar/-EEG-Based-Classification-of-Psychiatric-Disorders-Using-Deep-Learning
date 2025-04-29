import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from config import MODEL_CHECKPOINT

# Define your folders here
TEST_DIR = "tdata/"
NORMAL_DIR = os.path.join(TEST_DIR, "normal")
ABNORMAL_DIR = os.path.join(TEST_DIR, "abnormal")

# ------------------------
# EEG Classifier Definition
# ------------------------
class EEGClassifier(torch.nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (3, 7), padding=(1, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 4)),
            torch.nn.Conv2d(16, 32, (3, 5), padding=(1, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 4))
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.classifier(x)

# ------------------------
# EEG Dataset Loader
# ------------------------
class EEGFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        for label, cls in enumerate(["normal", "abnormal"]):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(cls_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = np.load(self.data[idx]).astype(np.float32)  # shape: (24, 30000)
        eeg = torch.tensor(eeg).unsqueeze(0)  # shape: (1, 24, 30000)
        label = self.labels[idx]
        return eeg, torch.tensor(label)

# ------------------------
# Confusion Matrix
# ------------------------
def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("[INFO] Saved confusion matrix to confusion_matrix.png")

# ------------------------
# Saliency Map
# ------------------------
def plot_saliency_map(model, dataloader, device, idx=0):
    model.eval()
    x, y = next(iter(dataloader))
    x = x[idx:idx+1].to(device)
    x.requires_grad_()
    output = model(x)
    score = output[0, output.argmax()]
    score.backward()
    saliency = x.grad.abs().squeeze().cpu().numpy()
    avg_saliency = saliency.mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_saliency)), avg_saliency)
    plt.title("Saliency Map (EEG Channel Importance)")
    plt.xlabel("Channel Index")
    plt.ylabel("Gradient Magnitude")
    plt.grid(True)
    plt.savefig("saliency_map.png")
    print("[INFO] Saved saliency map to saliency_map.png")

# ------------------------
# Class Activation Map (CAM)
# ------------------------
def plot_cam(model, dataloader, device, idx=0):
    class FeatureWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = None
            self.base_model = base_model
            self.conv_layers = base_model.conv
            self.pool = base_model.pool
            self.classifier = base_model.classifier

        def forward(self, x):
            for i, layer in enumerate(self.conv_layers):
                x = layer(x)
                if i == 3:  # After second conv
                    self.features = x
            pooled = self.pool(x)
            logits = self.classifier(pooled)
            return logits

    wrapped = FeatureWrapper(model).to(device)
    wrapped.eval()

    x, _ = next(iter(dataloader))
    x = x[idx:idx+1].to(device)
    x.requires_grad_()

    out = wrapped(x)
    out[0, out.argmax()].backward()

    features = wrapped.features.detach()
    grads = x.grad.detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features, dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)

    plt.figure(figsize=(10, 4))
    plt.imshow(cam, cmap="jet", aspect="auto")
    plt.title("Class Activation Map (CAM)")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Feature Channels")
    plt.savefig("cam_map.png")
    print("[INFO] Saved CAM to cam_map.png")

# ------------------------
# Main Evaluation
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))

    dataset = EEGFolderDataset(TEST_DIR)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("[INFO] Confusion Matrix")
    plot_confusion_matrix(model, test_loader, device)

    print("[INFO] Saliency Map")
    plot_saliency_map(model, test_loader, device, idx=0)

    print("[INFO] Class Activation Map (CAM)")
    plot_cam(model, test_loader, device, idx=0)
