import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models import SimpleUNet

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEG_MODEL_PATH = 'segmentation_model.pth'
OBS_MODEL_PATH = 'obstacle_model.pth'
FAULT_MODEL_PATH = 'fault_model.pth'

# Class Names (Hardcoded based on dataset inspection)
OBSTACLE_CLASSES = ['Background', 'Barrel', 'Boulder', 'Branch', 'IronRod', 'Jerrycan', 'Person']
FAULT_CLASSES = ['Defective', 'Non defective']

def load_models():
    print("Loading models...")
    
    # 1. Segmentation
    seg_model = SimpleUNet(3, 1)
    if os.path.exists(SEG_MODEL_PATH):
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {SEG_MODEL_PATH} not found. Running with random weights (for demo).")
    seg_model.to(DEVICE)
    seg_model.eval()
    
    # 2. Obstacle Detection
    obs_model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = obs_model.roi_heads.box_predictor.cls_score.in_features
    obs_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(OBSTACLE_CLASSES))
    
    if os.path.exists(OBS_MODEL_PATH):
        obs_model.load_state_dict(torch.load(OBS_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {OBS_MODEL_PATH} not found. Running with random weights (for demo density).")
        
    obs_model.to(DEVICE)
    obs_model.eval()
    
    # 3. Fault Classification
    fault_model = models.resnet18(weights=None)
    fault_model.fc = torch.nn.Linear(fault_model.fc.in_features, len(FAULT_CLASSES))
    
    if os.path.exists(FAULT_MODEL_PATH):
        fault_model.load_state_dict(torch.load(FAULT_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {FAULT_MODEL_PATH} not found. Running with random weights.")
        
    fault_model.to(DEVICE)
    fault_model.eval()
    
    return seg_model, obs_model, fault_model

def get_rail_roi(mask):
    """
    Finds the bounding rect of the rail mask to crop the 'track' for fault analysis.
    """
    points = cv2.findNonZero(mask)
    if points is None:
        return None # No rail found
    x, y, w, h = cv2.boundingRect(points)
    # Add some padding
    pad = 50
    h_img, w_img = mask.shape
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad)
    h = min(h_img - y, h + 2*pad)
    return (x, y, w, h)

def run_inference(image_path, models_tuple):
    seg_model, obs_model, fault_model = models_tuple
    
    # Load Image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error loading {image_path}")
        return
    
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # -----------------------
    # 1. Segmentation
    # -----------------------
    # Resize for model (256x256) then scale mask back
    seg_input = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])(img_rgb).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        seg_out = seg_model(seg_input)
        seg_out = torch.sigmoid(seg_out)
        seg_out = (seg_out > 0.5).float()
    
    # Upsample mask to original size
    mask = F.interpolate(seg_out, size=(h, w), mode='nearest').squeeze().cpu().numpy().astype(np.uint8)
    
    # -----------------------
    # 2. Obstacle Detection
    # -----------------------
    obs_input = T.ToTensor()(img_rgb).to(DEVICE)
    
    with torch.no_grad():
        obs_out = obs_model([obs_input])[0]
        
    boxes = obs_out['boxes'].cpu().numpy()
    labels = obs_out['labels'].cpu().numpy()
    scores = obs_out['scores'].cpu().numpy()
    
    # Filter by score
    score_thr = 0.5
    indices = scores > score_thr
    boxes = boxes[indices]
    labels = labels[indices]
    
    # Filter by Mask (Advanced Logic)
    valid_boxes = []
    valid_labels = []
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.astype(int)
        # Check intersection with rail mask
        # Create a mini-mask for the box
        box_mask = np.zeros_like(mask)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)
        
        intersection = np.logical_and(mask, box_mask)
        if np.sum(intersection) > 0: # If box touches rail
            valid_boxes.append(box)
            valid_labels.append(label)
        else:
            print(f"Filtered out obstacle {OBSTACLE_CLASSES[label]} at {box} - Not on rail")
            
    # -----------------------
    # 3. Fault Classification
    # -----------------------
    roi = get_rail_roi(mask)
    fault_status = "Unknown (No Rail)"
    color = (100, 100, 100)
    
    if roi:
        x, y, w, h_roi = roi
        track_crop = img_rgb[y:y+h_roi, x:x+w]
        
        if track_crop.size > 0:
            fault_input = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(track_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                fault_out = fault_model(fault_input)
                _, pred = torch.max(fault_out, 1)
                
            fault_status = FAULT_CLASSES[pred.item()]
            
            if fault_status == 'Defective':
                color = (0, 0, 255) # Red
            else:
                color = (0, 255, 0) # Green
    
    # -----------------------
    # 4. Visualization
    # -----------------------
    vis_img = original_img.copy()
    
    # Draw Mask Overlay (Red transparent)
    zeros = np.zeros_like(mask)
    mask_vis = cv2.merge([zeros, zeros, mask * 255])
    vis_img = cv2.addWeighted(vis_img, 1, mask_vis, 0.5, 0)
    
    # DrawBoxes
    for box, label in zip(valid_boxes, valid_labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Obstacles are Red
        if label < len(OBSTACLE_CLASSES):
            label_text = OBSTACLE_CLASSES[label]
        else:
            label_text = str(label)
        cv2.putText(vis_img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    # Draw Fault Status
    status_text = f"Track Status: {fault_status}"
    cv2.putText(vis_img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Save
    out_name = "output_inference.jpg"
    cv2.imwrite(out_name, vis_img)
    print(f"Inference complete. Saved to {out_name}")

if __name__ == "__main__":
    import os
    # Pick a random test image from one of the datasets
    # Let's use a segmentation image as it likely has a full track view
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'V1 UAV-RSOD_Dataset for Segmentation', '1 Images')
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    
    if test_images:
        test_img_path = os.path.join(TEST_DIR, test_images[0])
        print(f"Running inference on {test_img_path}")
        
        loaded_models = load_models()
        run_inference(test_img_path, loaded_models)
    else:
        print("No test images found.")
