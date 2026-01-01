import torch
import config

def gatekeeper(img_tensor, ae, backbone, threshold=config.GATEKEEPER_THRESHOLD):
    """
    Determines if an image is a valid brain MRI using autoencoder reconstruction error.
    
    Args:
        img_tensor: Single image tensor (C, H, W)
        ae: Trained autoencoder
        backbone: EfficientNet backbone for feature extraction
        threshold: Maximum reconstruction error allowed
        
    Returns:
        tuple: (is_valid, reconstruction_error)
    """
    ae.eval()
    backbone.eval()
    
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(config.DEVICE)
        emb = backbone(x)
        recon = ae(emb)
        error = torch.mean((emb - recon)**2).item()
    
    if error <= threshold:
        return True, error
    else:
        return False, error
