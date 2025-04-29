###################
# Author - Gurleen Kaur
# Contributors - 
# File - define_model.py
# This file defines the deep learning model architecture for EEG classification.
###################

import torch
import torch.nn as nn
from config import MODEL_CHECKPOINT


class EEGClassifier(nn.Module):
    def __init__(self, num_channels=24, num_classes=2):
        super(EEGClassifier, self).__init__()

        # Convolutional Layers for feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Adaptive pooling to ensure a fixed-size output before FC layers
        self.global_pool = nn.AdaptiveAvgPool2d((5, 5))  

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ensure correct shape: (batch, 1, channels, time)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)  
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
