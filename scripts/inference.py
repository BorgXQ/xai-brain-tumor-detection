"""
Inference script for single image prediction with OOD detection.
"""
import sys
sys.path.append('..')

import argparse
from PIL import Image
import config
from src.data.dataloader import get_data_loaders
from src.utils.model_loader import load_ensemble, load_gatekeeper
from src.models.ensemble import create_backbone
from src.inference.gatekeeper import gatekeeper
from src.inference.predict import ensemble_predict_mc

def main():
    parser = argparse.ArgumentParser(description='Brain tumor classification inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--threshold', type=float, default=config.GATEKEEPER_THRESHOLD,
                        help='OOD detection threshold')
    args = parser.parse_args()
    
    print("Loading models...")
    _, _, train_ds, _ = get_data_loaders()
    ensemble = load_ensemble(config.NUM_MODELS, config.NUM_CLASSES, config.DEVICE)
    ae = load_gatekeeper(config.DEVICE)
    backbone = create_backbone()
    
    print(f"\nProcessing image: {args.image}")
    
    try:
        img = Image.open(args.image).convert("RGB")
        img_t = config.tfm(img)
        
        # Gatekeeper check
        valid, recon_err = gatekeeper(img_t, ae, backbone, args.threshold)
        
        if valid:
            print(f"Image passed gatekeeper check (Reconstruction error: {recon_err:.4f}).")
            print("Proceeding to ensemble prediction.\n")
            ensemble_predict_mc(args.image, ensemble, train_ds, gradcam_model_idx=0)
        else:
            print(f"Image rejected by gatekeeper (Reconstruction error: {recon_err:.4f}).")
            print("Not a valid brain MRI.")
    
    except Exception as e:
        print(f"Error loading image: {e}")

if __name__ == "__main__":
    main()
