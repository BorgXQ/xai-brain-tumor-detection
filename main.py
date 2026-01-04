"""
Main entry point for the brain tumor classification system.
Provides a CLI interface for training, evaluation, and inference.
"""
import argparse
from PIL import Image

import config
from src.data.dataloader import get_data_loaders
from src.models.ensemble import create_ensemble, create_backbone
from src.models.autoencoder import EmbeddingAutoencoder
from src.training.train_ensemble import train_ensemble
from src.training.train_autoencoder import train_autoencoder
from src.utils.model_loader import load_ensemble, load_gatekeeper
from src.eval.evaluate import evaluate_ensemble
from src.inference.gatekeeper import gatekeeper
from src.inference.predict import ensemble_predict_mc

def train():
    """Train ensemble and autoencoder."""
    print("Loading data...")
    train_loader, _, _, _ = get_data_loaders()
    
    print("\n=== Training Ensemble ===")
    ensemble = create_ensemble(config.NUM_MODELS)
    train_ensemble(ensemble, train_loader)
    
    print("\n=== Training Autoencoder ===")
    backbone = create_backbone()
    ae = EmbeddingAutoencoder().to(config.DEVICE)
    train_autoencoder(ae, backbone, train_loader)
    
    print("\nTraining complete!")

def evaluate():
    """Evaluate ensemble on test set."""
    print("Loading data...")
    _, test_loader, _, _ = get_data_loaders()
    
    print("Loading ensemble models...")
    ensemble = load_ensemble(config.NUM_MODELS, config.NUM_CLASSES, config.DEVICE)
    
    print("""
███████╗██╗   ██╗ █████╗ ██╗        ███████╗████████╗ █████╗ ██████╗ ████████╗
██╔════╝██║   ██║██╔══██╗██║        ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝
█████╗  ██║   ██║███████║██║        ███████╗   ██║   ███████║██████╔╝   ██║   
██╔══╝  ╚██╗ ██╔╝██╔══██║██║        ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   
███████╗ ╚████╔╝ ██║  ██║███████╗   ███████║   ██║   ██║  ██║██║  ██║   ██║   
╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      
""")
    
    evaluate_ensemble(ensemble, test_loader)

def inference(image_path, threshold=config.GATEKEEPER_THRESHOLD):
    """Run inference on a single image."""
    print("Loading models...")
    ensemble = load_ensemble(config.NUM_MODELS, config.NUM_CLASSES, config.DEVICE)
    ae = load_gatekeeper(config.DEVICE)
    backbone = create_backbone()
    
    print(f"\nProcessing image: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = config.tfm(img)
        
        valid, recon_err = gatekeeper(img_t, ae, backbone, threshold)
        
        if valid:
            print(f"Image passed gatekeeper check (Reconstruction error: {recon_err:.4f}).")
            print("Proceeding to ensemble prediction.\n")
            ensemble_predict_mc(image_path, ensemble)
        else:
            print(f"Image rejected by gatekeeper (Reconstruction error: {recon_err:.4f}).")
            print("Not a valid brain MRI.")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py evaluate
  python main.py inference --image path/to/image.jpg
  python main.py inference --image path/to/image.jpg --threshold 0.15
        """
    )
    
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference'],
                        help='Mode to run: train, evaluate, or inference')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--threshold', type=float, default=config.GATEKEEPER_THRESHOLD,
                        help=f'OOD detection threshold (default: {config.GATEKEEPER_THRESHOLD})')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'inference':
        if not args.image:
            parser.error("inference mode requires --image argument")
        inference(args.image, args.threshold)

if __name__ == "__main__":
    main()
