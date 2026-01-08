import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. Segmentation Dataset (V1)
# ---------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with '1 Images' and '2 Annotations' folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, '1 Images')
        # Based on exploration, masks are in '2 Annotations/2.2 Masking/Rail Lines'
        # Check if '2.2 Masking' exists, else try finding where masks are.
        # The user's `list_dir` showed: ...\2 Annotations\2.2 Masking\Rail Lines\*.jpg
        self.mask_dir = os.path.join(root_dir, '2 Annotations', '2.2 Masking', 'Rail Lines')
        
        # Filter for images that exist in both folders
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        # Ensure corresponding mask exists
        self.valid_images = []
        for img in self.images:
            if os.path.exists(os.path.join(self.mask_dir, img)):
                self.valid_images.append(img)
        
        print(f"[SegmentationDataset] Found {len(self.valid_images)} valid image-mask pairs in {root_dir}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (It seems masks are JPGs in this dataset, likely black/white)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Resize mask to match image dimensions if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            # Note: For segmentation, we need to apply same transform to both.
            # Here we assume transform is simple resizing/normalization usually.
            # For complex augmentation, use albumentations (omitted for MVP simplicity)
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        # Convert to Tensor
        # If transform provided standard tensor conversion, use it. 
        # Otherwise manually convert if using vanilla transforms pipeline.
        
        # For simplicity in this script, let's return PIL/Numpy if no transform, 
        # or assume the user handles tensor conversion in the transform.
        # But to be safe for PyTorch DataLoader:
        
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        if not torch.is_tensor(mask):
            mask = transforms.ToTensor()(mask)

        return image, mask

# ---------------------------------------------------------
# 2. Obstacle Detection Dataset (V2)
# ---------------------------------------------------------
class ObstacleDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Load annotations
        # Path: .../images/train_labels.csv
        csv_path = os.path.join(root_dir, 'images', 'train_labels.csv')
        self.df = pd.read_csv(csv_path)
        
        self.image_dir = os.path.join(root_dir, 'images', 'train')
        self.image_ids = self.df['filename'].unique()
        
        # Map class names to IDs
        # We need a consistent map. Let's create one based on the dataset.
        self.classes = sorted(self.df['class'].unique())
        self.class_to_id = {cls_name: i+1 for i, cls_name in enumerate(self.classes)}
        # 0 is reserved for background
        
        print(f"[ObstacleDataset] Found {len(self.image_ids)} images with {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Get all boxes for this image
        records = self.df[self.df['filename'] == img_name]
        
        boxes = []
        labels = []
        
        for _, row in records.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_id[row['class']])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            # We assume transforms here is a function capable of handling 'target'
            # But standard torchvision transforms only handle image.
            # Taking a simpler approach: Just transform image, keep boxes as is (resizing strictly handled carefully)
            # For now, we assume ONLY ToTensor() is passed, no resizing that breaks boxes.
            image = self.transforms(image)

        return image, target

# ---------------------------------------------------------
# 3. Fault Classification Dataset
# ---------------------------------------------------------
# We can use standard ImageFolder if the structure is:
# Root
#   - Defective
#   - Non defective
from torchvision.datasets import ImageFolder

def get_fault_dataset(root_dir, transform=None):
    # Root dir should be .../Railway Track fault Detection Updated/Train
    dataset = ImageFolder(root=root_dir, transform=transform)
    print(f"[FaultDataset] Found {len(dataset)} images. Classes: {dataset.classes}")
    return dataset
