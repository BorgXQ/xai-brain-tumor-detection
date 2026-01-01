import os
import torch
import torch.nn as nn
import config

def train_model(model, train_loader, epochs):
    """
    Trains a single model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        epochs (int): Number of training epochs
        
    Returns:
        model: Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    
    return model

def train_ensemble(ensemble, train_loader):
    """
    Trains all models in the ensemble and saves their weights.
    
    Args:
        ensemble (list): List of models
        train_loader: Training data loader
    """
    os.makedirs(config.SAVED_MODELS_PATH, exist_ok=True)
    
    for i in range(config.NUM_MODELS):
        print(f"Training model {i+1}/{config.NUM_MODELS}")
        ensemble[i] = train_model(ensemble[i], train_loader, config.EPOCHS)
        
        torch.save(
            ensemble[i].state_dict(),
            f"{config.SAVED_MODELS_PATH}/efficientnet_b0_model_{i}.pth"
        )
        print(f"Model {i+1} saved.\n")
