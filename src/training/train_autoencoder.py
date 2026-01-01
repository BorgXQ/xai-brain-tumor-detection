import os
import torch
import torch.nn as nn
import config

def train_autoencoder(ae, backbone, train_loader):
    """
    Trains the autoencoder on embeddings from the backbone.
    
    Args:
        ae: Autoencoder model
        backbone: EfficientNet backbone for feature extraction
        train_loader: Training data loader
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    
    for epoch in range(config.AE_EPOCHS):
        ae.train()
        epoch_loss = 0
        
        for x, _ in train_loader:
            x = x.to(config.DEVICE)
            with torch.no_grad():
                emb = backbone(x)  # extract embeddings
            
            optimizer.zero_grad()
            recon = ae(emb)
            loss = criterion(recon, emb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config.AE_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    os.makedirs(config.SAVED_MODELS_PATH, exist_ok=True)
    torch.save(ae.state_dict(), f"{config.SAVED_MODELS_PATH}/embedding_autoencoder.pth")
    print("Autoencoder saved.")
