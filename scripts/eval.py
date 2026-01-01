"""
Evaluation script for the ensemble model.
"""
import sys
sys.path.append('..')

import config
from src.data.dataloader import get_data_loaders
from src.utils.model_loader import load_ensemble
from src.eval.evaluate import evaluate_ensemble

def main():
    print("Loading data...")
    train_loader, test_loader, train_ds, test_ds = get_data_loaders()
    
    print("Loading ensemble models...")
    ensemble = load_ensemble(config.NUM_MODELS, config.NUM_CLASSES, config.DEVICE)
    
    print("\n=== Evaluating Ensemble ===")
    evaluate_ensemble(ensemble, test_loader, test_ds)

if __name__ == "__main__":
    main()
