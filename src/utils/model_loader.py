import torch
import torch.nn as nn
from torchvision import models
import config
from src.models.autoencoder import EmbeddingAutoencoder

def load_ensemble(num_models, num_classes, device, model_dir=config.SAVED_MODELS_PATH):
    """
    Loads trained ensemble models from disk.
    
    Args:
        num_models (int): Number of models in ensemble
        num_classes (int): Number of output classes
        device (str): Device to load models on
        model_dir (str): Directory containing saved models
        
    Returns:
        list: List of loaded models
    """
    ensemble = []

    for i in range(num_models):
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        model.load_state_dict(
            torch.load(
                f"{model_dir}/efficientnet_b0_model_{i}.pth",
                map_location=device
            )
        )

        model.to(device)
        model.eval()
        ensemble.append(model)

    print(f"Loaded {num_models} models from {model_dir}")
    return ensemble

def load_gatekeeper(device, model_dir=config.SAVED_MODELS_PATH):
    """
    Loads trained autoencoder gatekeeper from disk.
    
    Args:
        device (str): Device to load model on
        model_dir (str): Directory containing saved model
        
    Returns:
        EmbeddingAutoencoder: Loaded autoencoder model
    """
    ae = EmbeddingAutoencoder().to(device)
    ae.load_state_dict(
        torch.load(
            f"{model_dir}/embedding_autoencoder.pth",
            map_location=device
        )
    )
    ae.eval()
    print("Loaded embedding autoencoder gatekeeper.")
    return ae
