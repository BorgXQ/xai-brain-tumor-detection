import cv2
import config

class GradCAM:
    """
    Grad-CAM for model explainability.
    """
    def __init__(self, model, layer):
        self.grad = None
        self.act = None
        self.model = model
        layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))
        layer.register_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        score = self.model(x)[0, class_idx]
        score.backward()

        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.act).sum(dim=1).relu()
        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))
        return cam / (cam.max() + 1e-8)
