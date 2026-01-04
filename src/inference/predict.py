import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import config
from src.inference.gradcam import GradCAM

def enable_mc_dropout(model):
    """
    Enables MC dropout for uncertainty estimation.
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def ensemble_predict_mc(img_path, ensemble, gradcam_model_idx=0, mc_passes=config.MC_PASSES):
    """
    Performs ensemble prediction with MC dropout for uncertainty estimation.
    
    Args:
        img_path (str): Path to image
        ensemble (list): List of trained models
        gradcam_model_idx (int): Which model to use for Grad-CAM
        mc_passes (int): Number of MC dropout passes
    """
    img = Image.open(img_path).convert("RGB")
    x = config.tfm(img).unsqueeze(0).to(config.DEVICE)

    all_probs = []
    votes = {class_name: 0 for class_name in config.CLASS_NAMES}

    for model_i in ensemble:
        model_i.eval()
        enable_mc_dropout(model_i)  # keep ONLY dropout active
        
        probs_list = []
        model_votes = []
        for _ in range(mc_passes):
            logits = model_i(x)
            probs_list.append(F.softmax(logits, dim=1))
            pred_class = logits.argmax(dim=1).item()
            model_votes.append(pred_class)
        
        majority_vote = max(set(model_votes), key=model_votes.count)
        votes[config.CLASS_NAMES[majority_vote]] += 1

        probs_stack = torch.stack(probs_list)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        all_probs.append((mean_probs, std_probs))

    # Stack ensemble results
    mean_probs_stack = torch.stack([p[0] for p in all_probs])
    std_probs_stack = torch.stack([p[1] for p in all_probs])

    ensemble_mean = mean_probs_stack.mean(dim=0)
    ensemble_std = std_probs_stack.mean(dim=0)

    pred_class = ensemble_mean.argmax(dim=1).item()
    pred_conf = ensemble_mean.max().item()
    pred_uncertainty = ensemble_std.max().item()  # epistemic uncertainty

    class_p_pairs = list(zip(config.CLASS_NAMES, ensemble_mean[0].tolist()))
    sorted_pairs = sorted(class_p_pairs, key=lambda x: x[1], reverse=True)

    print("Model predictive distribution:")
    for class_name, p in sorted_pairs:
        if p < 0.01:
            print(f"{class_name}: <0.01")
        else:
            print(f"{class_name}: {p: .2f}")
    print("\n")

    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    print("Ensemble votes (argmax decisions):")
    for class_name, count in sorted_votes:
        print(f"{class_name}: {count}/{len(ensemble)}")
    print("\n")
    
    if pred_conf >= 0.90:
        risk = "HIGH confidence"
    elif pred_conf >= 0.80:
        risk = "MEDIUM confidence"
    else:
        risk = "LOW confidence"

    print(f"Predicted: {config.CLASS_NAMES[pred_class]} ({risk}: {pred_conf*100:.2f}%)")
    print(f"Ensemble uncertainty (std across models): {pred_uncertainty:.4f}")
    
    # Grad-CAM visualization
    gradcam_model = ensemble[gradcam_model_idx]
    gradcam_model.eval()
    gradcam = GradCAM(gradcam_model, gradcam_model.features[-1])
    cam = gradcam(x, pred_class)
    img_np = np.array(img.resize((config.IMG_SIZE, config.IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    cv2.imshow("Grad-CAM", overlay)
    cv2.waitKey(0)
