import torch.nn as nn
from torchvision import models
import config

def create_efficientnet_model():
    """
    Creates a single EfficientNet-B0 model with custom classifier.
    
    Returns:
        nn.Module: Modified EfficientNet-B0 model
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, config.NUM_CLASSES)
    )
    model.to(config.DEVICE)
    return model

def create_ensemble(num_models):
    """
    Creates an ensemble of EfficientNet-B0 models.
    
    Args:
        num_models (int): Number of models in the ensemble
        
    Returns:
        list: List of EfficientNet-B0 models
    """
    ensemble = []
    for _ in range(num_models):
        model_i = create_efficientnet_model()
        ensemble.append(model_i)
    return ensemble

def create_backbone():
    """
    Creates EfficientNet-B0 backbone for feature extraction (for autoencoder).
    
    Returns:
        nn.Module: EfficientNet-B0 with identity classifier
    """
    backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
    backbone.classifier = nn.Identity()
    backbone.to(config.DEVICE)
    backbone.eval()
    return backbone
