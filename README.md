# Explainable Brain Tumor Classification System

A deep learning system for classifying human brain MRI images into four categories: glioma, meningioma, pituitary tumor, or no tumor. Uses a deep ensemble of EfficientNet-B0 models with Monte Carlo dropout for uncertainty estimation and an autoencoder-based out-of-distribution (OOD) detector.

## Features

- **Deep Ensemble**: 5 EfficientNet-B0 models for robust predictions
- **OOD Detection**: Autoencoder gatekeeper to reject invalid inputs
- **Uncertainty Estimation**: MC Dropout for epistemic uncertainty
- **Explainability**: Grad-CAM visualizations
- **4-Class Classification**: glioma, meningioma, pituitary, or no tumor

## Project Structure

```
brain_tumor_classifier/
├── config.py                     # Configuration and hyperparameters
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/
│   ├── models/
│   │   ├── ensemble.py           # EfficientNet ensemble
│   │   └── autoencoder.py        # OOD detector
│   │
│   ├── data/
│   │   └── dataloader.py         # Data loading utilities
│   │
│   ├── training/
│   │   ├── train_ensemble.py     # Ensemble training
│   │   └── train_autoencoder.py  # Autoencoder training
│   │
│   ├── evaluation/
│   │   └── evaluate.py           # Model evaluation
│   │
│   ├── inference/
│   │   ├── predict.py            # Prediction with MC dropout
│   │   ├── gradcam.py            # Grad-CAM visualization
│   │   └── gatekeeper.py         # OOD detection
│   │
│   └── utils/
│       ├── model_loader.py       # Load saved models
│       └── visualization.py      # Visualization utilities
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Inference script
│
├── saved_models/                 # Trained model weights
└── dataset/
    ├── Training/
    └── Testing/
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

Download the dataset from the following [link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

Organize the files as follows:

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Usage

### Training

Train both the ensemble and autoencoder:

```bash
python main.py train
```

### Evaluation

Evaluate the trained ensemble on the test set:

```bash
python main.py evaluate
```

### Inference

Run inference on a single image:

```bash
python main.py inference --image path/to/image.jpg
```

With custom OOD threshold:

```bash
python main.py inference --image path/to/image.jpg --threshold 0.15
```

## Configuration

Edit `config.py` to modify:

- `NUM_MODELS`: Number of ensemble models (default: 5)
- `EPOCHS`: Training epochs for ensemble (default: 5)
- `AE_EPOCHS`: Training epochs for autoencoder (default: 10)
- `BATCH_SIZE`: Batch size (default: 16)
- `IMG_SIZE`: Input image size (default: 256)
- `GATEKEEPER_THRESHOLD`: OOD detection threshold (default: 0.1)
- `MC_PASSES`: Monte Carlo dropout passes (default: 10)

## Model Architecture

### Ensemble
- 5 independent EfficientNet-B0 models
- Pre-trained on ImageNet
- Custom classifier with dropout (0.5)
- 4 output classes

### Autoencoder (Gatekeeper)
- Input: 1280-dim EfficientNet embeddings
- Latent: 256 dimensions
- Reconstruction-based OOD detection

## Output

The system provides:
1. Predicted class with confidence level
2. Uncertainty estimate (epistemic)
3. Grad-CAM visualization showing regions of focus
4. OOD detection to reject invalid inputs

## License

MIT

## Citation

If you use this code, please cite appropriately.