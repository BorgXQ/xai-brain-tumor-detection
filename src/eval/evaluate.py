import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
import config
from src.utils.visualization import plot_confusion

def evaluate_ensemble(ensemble, test_loader, test_ds):
    """
    Evaluates the ensemble on test data.
    
    Args:
        ensemble (list): List of trained models
        test_loader: Test data loader
        test_ds: Test dataset (for class names)
    """
    y_true, y_pred = [], []

    for x, y in test_loader:
        x = x.to(config.DEVICE)
        probs_list = []
        
        for model_i in ensemble:
            model_i.eval()
            with torch.no_grad():
                logits = model_i(x)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)
        
        probs_stack = torch.stack(probs_list)
        mean_probs = probs_stack.mean(dim=0)
        preds = mean_probs.argmax(dim=1).cpu()

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

    print("Accuracy:", accuracy_score(y_true, y_pred))

    plot_confusion(y_true, y_pred, test_ds.classes)

    print(
        "\nClassification Report:\n",
        classification_report(y_true, y_pred, target_names=test_ds.classes)
    )
