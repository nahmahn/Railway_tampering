import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import transforms
from dataset import SegmentationDataset, ObstacleDataset, get_fault_dataset

def visualize_segmentation(dataset, count=3):
    samples = []
    indices = np.random.choice(len(dataset), count, replace=False)
    for idx in indices:
        img, mask = dataset[idx] # Returns Tesnors
        
        # Convert to numpy
        img_np = img.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
        mask_np = mask.permute(1, 2, 0).numpy()
        
        # Overlay
        # Mask is 1-channel, make it red RGB
        mask_rgb = np.zeros_like(img_np)
        mask_rgb[:, :, 0] = mask_np[:, :, 0] # Red channel
        
        # Blend
        overlay = cv2.addWeighted(img_np, 0.7, mask_rgb, 0.3, 0)
        samples.append(overlay)
    return samples

def visualize_detection(dataset, count=3):
    samples = []
    indices = np.random.choice(len(dataset), count, replace=False)
    
    # We need to map IDs back to class names
    id_to_class = {v: k for k, v in dataset.class_to_id.items()}
    
    for idx in indices:
        img, target = dataset[idx]
        img_np = img.permute(1, 2, 0).contiguous().numpy().copy()
        
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            name = id_to_class.get(label, str(label))
            cv2.putText(img_np, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        samples.append(img_np)
    return samples

def visualize_classification(dataset, count=3):
    samples = []
    indices = np.random.choice(len(dataset), count, replace=False)
    
    classes = dataset.classes
    
    for idx in indices:
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        
        class_name = classes[label]
        
        # Just return the numpy array and label
        samples.append((img_np, class_name))
    return samples

def main():
    # Paths
    # IMPORTANT: Adjust these to match YOUR workspace
    # Since this script is now in 'backend/', ROOT is '..' relative to this script, 
    # but we are using absolute path for safety or relative.
    # Let's use absolute path based on the user's known workspace or relative to script.
    
    # Using relative path from this script (backend/visualize_data.py) to root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(SCRIPT_DIR) # hack4Delhi/
    DATA_DIR = os.path.join(ROOT, "data")
    
    V1_DIR = os.path.join(DATA_DIR, "V1 UAV-RSOD_Dataset for Segmentation")
    V2_DIR = os.path.join(DATA_DIR, "V2 UAV-RSOD_Dataset for Obstacle Detection")
    # Fault dataset has Train/Test/Val split
    FAULT_DIR = os.path.join(DATA_DIR, "Railway Track fault Detection Updated", "Train")
    
    print(f"Data Root: {DATA_DIR}")
    print("Loading datasets...")
    
    # Transforms
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # 1. Seg
    ds_seg = SegmentationDataset(V1_DIR, transform=None) 
    
    # 2. Det
    ds_det = ObstacleDataset(V2_DIR, transforms=transforms.Compose([transforms.ToTensor()]))
    
    # 3. Fault
    ds_fault = get_fault_dataset(FAULT_DIR, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))
    
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Dataset Visualization Check")
    
    # Row 1: Segmentation
    # We need to manually resize for plot consistency if images are huge
    imgs_seg = visualize_segmentation(ds_seg, count=3)
    for i, img in enumerate(imgs_seg):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Segmentation Sample {i+1}")
        axes[0, i].axis('off')
        
    # Row 2: Detection
    imgs_det = visualize_detection(ds_det, count=3)
    for i, img in enumerate(imgs_det):
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Detection Sample {i+1}")
        axes[1, i].axis('off')
        
    # Row 3: Fault
    data_fault = visualize_classification(ds_fault, count=3)
    for i, (img, label) in enumerate(data_fault):
        axes[2, i].imshow(img)
        axes[2, i].set_title(f"Fault: {label}")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    
    # Save directly to frontend public folder so it's visible in UI
    output_path = os.path.join(ROOT, 'frontend', 'public', 'mock-images', 'dataset_vis.jpg')
    plt.savefig(output_path)
    print(f"Saved visualization to '{output_path}'")

if __name__ == "__main__":
    main()
