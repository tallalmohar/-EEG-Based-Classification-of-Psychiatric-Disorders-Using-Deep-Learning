import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from config import RESULTS_DIR  # Make sure this points to your config file

# Set the save directory for Grad-CAM results
GRAD_CAM_DIR = os.path.join(RESULTS_DIR, "GradCAM")
os.makedirs(GRAD_CAM_DIR, exist_ok=True)

def get_grad_cam(model, eeg_data, class_idx, layer_name='conv2'):
    """
    Computes Grad-CAM heatmap for an EEG sample using a target convolutional layer.
    Returns: grad_cam (channels x time), output, predicted class
    """
    device = next(model.parameters()).device
    model.eval()

    features, gradients = [], []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    target_layer = dict(model.named_modules())[layer_name]
    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    eeg_data = eeg_data.unsqueeze(0).to(device).requires_grad_(True)
    output = model(eeg_data)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()

    # Remove hooks
    f_handle.remove()
    b_handle.remove()

    # Get features and gradients
    conv_output = features[0].detach().cpu().numpy().squeeze()  # shape: (C, T, T)
    grad_output = gradients[0].detach().cpu().numpy().squeeze()

    # Grad-CAM: weight each channel by the avg gradient
    weights = np.mean(grad_output, axis=(1, 2))  # average across time
    grad_cam = np.zeros_like(conv_output[0])

    for i, w in enumerate(weights):
        grad_cam += w * conv_output[i]

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-10)

    # Upsample to 30,000 time steps
    grad_cam_tensor = torch.tensor(grad_cam).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    upsampled = F.interpolate(grad_cam_tensor, size=(grad_cam.shape[0], 30000), mode='bilinear', align_corners=False)
    grad_cam_upsampled = upsampled.squeeze().numpy()  # (channels, 30000)

    return grad_cam_upsampled, output.detach().cpu().numpy(), pred_class


def visualize_grad_cam(model, eeg_data, class_idx, layer_name='conv2'):
    """
    Visualizes and saves the Grad-CAM as a 2D heatmap (channels × time).
    """
    grad_cam, pred_score, pred_class = get_grad_cam(model, eeg_data, class_idx, layer_name)

    save_path = os.path.join(GRAD_CAM_DIR, f"grad_cam_class{pred_class}.png")

    plt.figure(figsize=(14, 6))
    plt.imshow(grad_cam, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label="Activation")
    plt.xlabel("Time Steps - 30,000)")
    plt.ylabel("Feature Maps / Channels")
    plt.title(f"Grad-CAM Heatmap for Class {pred_class} (Score: {pred_score[0, class_idx]:.2f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Grad-CAM heatmap saved at {save_path}")
