from torch.utils.data import DataLoader
from torchvision import datasets
import config

def get_data_loaders():
    """
    Creates and returns train and test data loaders.
    
    Returns:
        tuple: (train_loader, test_loader, train_ds, test_ds)
    """
    train_ds = datasets.ImageFolder(config.TRAIN_PATH, transform=config.tfm)
    test_ds = datasets.ImageFolder(config.TEST_PATH, transform=config.tfm)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)
    
    return train_loader, test_loader, train_ds, test_ds
