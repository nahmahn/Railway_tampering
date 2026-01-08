import torch
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os

from dataset import ObstacleDataset

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_obstacle(num_epochs=10, batch_size=2, learning_rate=0.005):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'V2 UAV-RSOD_Dataset for Obstacle Detection')
    
    # For detection, we usually just convert to tensor. 
    # Resizing is tricky because valid boxes need to be scaled too. 
    # For MVP, we pass images as-is (dataset returns standard size 1920x1080? or variable).
    # If variable, batching requires them to be same size OR use a model that handles variable sizes.
    # Faster R-CNN handles variable sizes, but batch_size > 1 requires careful collation usually 
    # unless we resize.
    # The dataset.py returns (image, target). 
    # Let's rely on the default torchvision transforms.ToTensor() passed in dataset.py inside the script?
    # Actually dataset.py accepts a transform.
    
    dataset = ObstacleDataset(ROOT_DIR, transforms=torchvision.transforms.ToTensor())
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    
    # 2. Model
    # Classes: Background + Dataset Classes
    num_classes = len(dataset.classes) + 1 # +1 for background
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    print(f"Starting Training for {num_classes} classes...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Tqdm loop
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass (During training, model returns loss dict)
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())
            
        lr_scheduler.step()
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(data_loader):.4f}")
        
    save_path = "obstacle_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_obstacle()
