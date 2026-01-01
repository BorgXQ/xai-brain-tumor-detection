import torch.nn as nn
import config

class EmbeddingAutoencoder(nn.Module):
    """
    Autoencoder for detecting out-of-distribution (OOD) samples.
    Trained on EfficientNet-B0 embeddings.
    """
    def __init__(self, input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
