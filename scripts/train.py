"""
Training script for ensemble and autoencoder.
"""
import sys
sys.path.append('..')

import config
from src.data.dataloader import get_data_loaders
from src.models.ensemble import create_ensemble, create_backbone
from src.models.autoencoder import EmbeddingAutoencoder
from src.training.train_ensemble import train_ensemble
from src.training.train_autoencoder import train_autoencoder

def main():
    print("Loading data...")
    train_loader, _, _, _ = get_data_loaders()
    
    # Train ensemble
    print("\n=== Training Ensemble ===")
    ensemble = create_ensemble(config.NUM_MODELS)
    train_ensemble(ensemble, train_loader)
    
    # Train autoencoder
    print("\n=== Training Autoencoder ===")
    backbone = create_backbone()
    ae = EmbeddingAutoencoder().to(config.DEVICE)
    train_autoencoder(ae, backbone, train_loader)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
