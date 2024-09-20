import torch
import cv2
from ultralytics import YOLO
from models import GaussianHead

# Load the trained model
model = YOLO('path/to/your/trained_model.pt')
model.model.model[-1] = GaussianHead(nc=80, anchors=model.model.model[-1].anchors)
model.model.to(device)

# Load an image
image_path = 'path/to/image.jpg'
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255

# Inference
model.model.eval()
with torch.no_grad():
    preds = model.model(img_tensor)
    # Process predictions to extract boxes and uncertainties
    # ...

# Display or save the results
# ...
