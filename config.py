import torch
from torchvision import transforms

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters
NUM_CLASSES = 4
NUM_MODELS = 10
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 5
AE_EPOCHS = 10

# Class names
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Autoencoder configuration
INPUT_DIM = 1280
LATENT_DIM = 256

# Paths
DATASET_PATH = "./dataset"
TRAIN_PATH = "./dataset/Training"
TEST_PATH = "./dataset/Testing"
SAVED_MODELS_PATH = "./saved_models"
SAVED_PLOTS_PATH = "./saved_eval_plots"

# Image transformations
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gatekeeper threshold
GATEKEEPER_THRESHOLD = 0.1

# MC Dropout settings
MC_PASSES = 10