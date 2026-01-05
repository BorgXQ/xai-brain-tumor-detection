import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
import config
import matplotlib.pyplot as plt
from src.utils.visualization import plot_confusion, plot_dashboard

def evaluate_ensemble(ensemble, test_loader):
    """
    Evaluates the ensemble on test data.
    
    Args:
        ensemble (list): List of trained models
        test_loader: Test data loader
    """
    y_true, y_pred = [], []
    all_conf = []

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
        confs, preds = mean_probs.max(dim=1)

        preds_np = preds.cpu().numpy()
        y_np = y.numpy()
        confs_np = confs.cpu().numpy()

        y_true.extend(y_np)
        y_pred.extend(preds_np)
        all_conf.extend(confs_np)

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("\nAccuracy at various confidence thresholds:")
    for threshold in [0.99, 0.90, 0.80, 0.70, 0.60, 0.50]:
        high_conf_mask = np.array(all_conf) >= threshold
        if high_conf_mask.any():
            total = high_conf_mask.sum()
            correct = (np.array(y_pred)[high_conf_mask] == np.array(y_true)[high_conf_mask]).sum()
            accuracy = correct / total if total > 0 else 0
            percentage = total / len(y_true) * 100
            print(f"  - At ≥{threshold:.2f}: {percentage:.1f}% of predictions, {accuracy:.1%} correct ({correct}/{total})")

    print(
        "\nClassification Report:\n",
        classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    )

    plot_confusion(y_true, y_pred, config.CLASS_NAMES)

    all_conf = np.array(all_conf)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = (y_pred == y_true)

    plot_dashboard(all_conf, correct, y_true, y_pred)
