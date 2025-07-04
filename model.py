import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from gradcam import GradCAM
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ------------------ Plain EfficientNet Model ------------------

class EfficientNetSimple(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetSimple, self).__init__()
        base = efficientnet_b0(pretrained=False)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ------------------ Prediction Function ------------------

MODEL_PATH = "best_model_eff_scratch.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Myocardial Infarction", "History of MI", "Abnormal ECG", "Normal"]

# ------------------ XAI Attribution Methods ------------------

def generate_ig_overlay(model, input_tensor, original_image, target_class):
    model.zero_grad()
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=50)
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))  # CHW -> HWC
    attributions = np.mean(np.abs(attributions), axis=2)

    # Normalize to [0, 255]
    attributions -= attributions.min()
    attributions /= (attributions.max() + 1e-8)
    attributions = np.uint8(attributions * 255)

    # Apply colormap
    heatmap = cv2.applyColorMap(attributions, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    return overlay

# ------------------ Textual XAI ------------------

def get_hotspot_region(cam):
    heat = cam.squeeze()
    y, x = np.unravel_index(np.argmax(heat), heat.shape)

    if x < 75:
        region = "left region"
    elif x > 150:
        region = "right region"
    else:
        region = "central region"
    return region

XAI_TEMPLATES = {
    "Myocardial Infarction": "The model focused on ST elevation in the {region}, commonly associated with myocardial infarction.",
    "History of MI": "The model identified signal flattening in the {region}, suggestive of prior infarction.",
    "Abnormal ECG": "The model detected waveform irregularities in the {region}, indicating potential abnormalities.",
    "Normal": "The model saw balanced signals across the {region}, consistent with normal ECG patterns."
}

def generate_textual_xai(cam, predicted_label):
    region = get_hotspot_region(cam)
    template = XAI_TEMPLATES.get(predicted_label, "No explanation available.")
    return template.format(region=region)

# ------------------ Main Prediction Function ------------------

def predict_ecg_class(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    # Load model
    model = EfficientNetSimple(num_classes=4).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Predict
    output = model(input_tensor)
    pred_idx = torch.argmax(output, dim=1).item()
    label = CLASS_NAMES[pred_idx]

    # Grad-CAM
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    target_layer = model.features[-1]  # Corrected reference
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor)

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    gradcam_overlay = np.uint8(cam_heatmap * 0.4 + original_rgb * 0.6)

    # ------------------ Integrated Gradients ------------------
    ig_overlay = generate_ig_overlay(model, input_tensor, original_rgb, pred_idx)

    textual_explanation = generate_textual_xai(cam, label)

    return label, gradcam_overlay, ig_overlay, textual_explanation

    # ------------------ Integrated Gradients ------------------
    ig_overlay = generate_ig_overlay(model, input_tensor, original_rgb, pred_idx)
    textual_explanation = generate_textual_xai(cam, label)

    return label, gradcam_overlay, ig_overlay, textual_explanation
