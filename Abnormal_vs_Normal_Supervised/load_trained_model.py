import torch
from define_model import EEGClassifier  
from config import MODEL_CHECKPOINT

def load_trained_model(model_path=MODEL_CHECKPOINT):
    """Loads the trained EEG classification model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGClassifier().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    print("[INFO] Model Loaded Successfully!")
    return model, device
