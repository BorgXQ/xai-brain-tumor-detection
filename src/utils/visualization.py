import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import config

def plot_confusion(y_true, y_pred, class_names):
    """
    Prints and plots confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(cm)
    print()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_confidence_distribution(ax, all_conf):
    """
    Plots histogram of prediction confidences.

    Args:
        ax: Matplotlib axis to plot on
        all_conf: Array of confidence scores
    """
    sns.histplot(
        all_conf,
        bins=50,
        ax=ax,
        edgecolor="black",
        color=sns.color_palette()[0],
        alpha=0.8,
        stat="density"
    )
    mean_conf = all_conf.mean()
    ax.axvline(mean_conf, color="red", linestyle="--", linewidth=2.5, label=f"Mean: {mean_conf:.3f}")
    ax.set_xlabel("Confidence Score", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Density (log scale)", fontsize=12, fontweight="semibold")
    ax.set_yscale("log")
    ax.set_title("Distribution of Prediction Confidences", fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper left")
    ax.grid(False)
    ax.set_facecolor("#f8f9fa")
    
def plot_reliability_diagram(ax, all_conf, correct):
    """
    Plots reliability diagram (calibration curve).
    
    Args:
        ax: Matplotlib axis to plot on
        all_conf: Array of confidence scores
        correct: Boolean array indicating correct predictions
    """
    bins = np.linspace(0.5, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (all_conf >= bins[i]) & (all_conf < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_counts.append(0)

    ax.plot(
        bin_centers,
        bin_accs,
        marker="o",
        markersize=10,
        linewidth=3.0,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="Model Calibration",
        color=sns.color_palette()[1]
    )

    for x, y, count in zip(bin_centers, bin_accs, bin_counts):
        if count > 0:
            ax.text(
                x, y - 0.05,
                f"N={count}",
                ha="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
            )

    ece = 0
    total_samples = len(all_conf)
    for i in range(len(bin_centers)):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * abs(bin_accs[i] - bin_centers[i])
    ax.text(
        0.51, 0.9,
        f"ECE: {ece:.4f}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8)
    )

    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="Perfect Calibration", alpha=0.7)

    ax.set_xlabel("Predicted Confidence", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Actual Accuracy", fontsize=12, fontweight="semibold")
    ax.set_title("Reliability Diagram (Calibration Curve)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper left")
    ax.grid(False)
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([0.2, 1.05])
    ax.set_facecolor("#f8f9fa")

def plot_conf_vs_error_scatter(ax, all_conf, correct):
    """
    Plots scatter of confidence vs prediction correctness.

    Args:
        ax: Matplotlib axis to plot on
        all_conf: Array of confidence scores
        correct: Boolean array indicating correct predictions
    """
    jitter = np.random.normal(0, 0.04, size=len(correct))
    scatter_y = correct.astype(int) + jitter

    scatter = ax.scatter(
        all_conf,
        scatter_y,
        c=scatter_y,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        s=40
    )

    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5)

    ax.set_xlabel("Confidence Score", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=12, fontweight="semibold")
    ax.set_title("Confidence vs Prediction Correctness", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim([-0.5, 1.5])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.grid(False)
    ax.set_facecolor("#f8f9fa")

def plot_conf_by_cls(ax, all_conf, y_true, y_pred):
    """
    Plots average confidence and accuracy by class.
    
    Args:
        ax: Matplotlib axis to plot on
        all_conf: Array of confidence scores
        y_true: True labels
        y_pred: Predicted labels
    """
    class_confidences = []
    class_accuracies = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        cls_mask = (y_pred == class_idx)
        if cls_mask.sum() > 0:
            class_confidences.append(all_conf[cls_mask].mean())
            correct_cls = ((y_pred == class_idx) & (y_true == class_idx)).sum()
            total_cls = (y_pred == class_idx).sum()
            class_accuracies.append(correct_cls / total_cls if total_cls > 0 else 0)
        else:
            class_confidences.append(0)
            class_accuracies.append(0)

    x = np.arange(len(config.CLASS_NAMES))
    width = 0.35

    bars1 = ax.bar(x - width/2, class_confidences, width, label="Avg Confidence",
                    color=sns.color_palette()[0], alpha=0.8, edgecolor="black")
    bars2 = ax.bar(x + width/2, class_accuracies, width, label="Accuracy",
                    color=sns.color_palette()[2], alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    # Style the plot
    ax.set_xlabel("Class", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Score", fontsize=12, fontweight="semibold")
    ax.set_title("Confidence and Accuracy by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45 if len(config.CLASS_NAMES) > 5 else 0)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(False)
    ax.set_ylim([0, 1.1])
    ax.set_facecolor("#f8f9fa")

def plot_dashboard(all_conf, correct, y_true, y_pred):
    """
    Plots a dashboard with multiple confidence analysis visualizations.

    Args:
        all_conf: Array of confidence scores
        correct: Boolean array indicating correct predictions
        y_true: True labels
        y_pred: Predicted labels
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Confidence Analysis Dashboard", fontsize=16, fontweight="bold", y=1.02)

    # Confidence Distribution Histogram
    ax1 = axes[0, 0]
    plot_confidence_distribution(ax1, all_conf)

    # Reliability Diagram
    ax2 = axes[0, 1]
    plot_reliability_diagram(ax2, all_conf, correct)

    # Confidence vs Error Scatter Plot
    ax3 = axes[1, 0]
    plot_conf_vs_error_scatter(ax3, all_conf, correct)

    # Confidence by Class
    ax4 = axes[1, 1]
    plot_conf_by_cls(ax4, all_conf, y_true, y_pred)

    plt.tight_layout()
    plt.show()
