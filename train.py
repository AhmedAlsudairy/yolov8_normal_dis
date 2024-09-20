import torch
from ultralytics import YOLO
from models import GaussianHead
from utils.gaussian_loss import gaussian_yolo_loss
from torch.utils.data import DataLoader
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.engine.validator import Valer

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Replace the detection head with the custom Gaussian head
anchors = model.model.model[-1].anchors  # Get anchors from the original model
nc = model.model.model[-1].nc  # Number of classes
model.model.model[-1] = GaussianHead(nc=nc, anchors=anchors)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.model.to(device)

# Configuration
cfg = DEFAULT_CFG
cfg['data'] = 'datasets/your_dataset/data.yaml'
cfg['imgsz'] = 640
cfg['batch'] = 16
cfg['epochs'] = 100
cfg['device'] = device

# Build dataloader
train_loader = build_dataloader(cfg, imgsz=cfg['imgsz'], batch_size=cfg['batch'], rank=-1, mode='train')

# Define optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

# Training loop
for epoch in range(cfg['epochs']):
    model.model.train()
    for batch_idx, batch in enumerate(train_loader):
        imgs = batch[0].to(device, non_blocking=True).float() / 255
        targets = batch[1].to(device)
        
        optimizer.zero_grad()
        preds = model.model(imgs)
        loss = gaussian_yolo_loss(preds, targets, device)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{cfg["epochs"]}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Optional: Validate after each epoch
    # val_metrics = Valer(model.model, cfg, dataloader=val_loader).validate()
