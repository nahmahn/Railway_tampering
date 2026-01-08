import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from tqdm import tqdm

from dataset import SegmentationDataset
from models import SimpleUNet

def train_segmentation(num_epochs=10, batch_size=4, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'V1 UAV-RSOD_Dataset for Segmentation')
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # Downsize for faster training
        transforms.ToTensor()
    ])
    
    full_dataset = SegmentationDataset(ROOT_DIR, transform=None)
    
    # Apply Resize here manually via collate or just resize in Dataset?
    # Dataset.py applies transform to both image and mask.
    # But wait, dataset.py expects a transform that takes (image=, mask=).
    # Standard torchvision transforms don't do that. 
    # Let's fix dataset.py or handle it here. 
    # The SIMPLEST way for this script: modify Dataset to accept standard transform or handle resizing inside.
    
    # Actually, in dataset.py I wrote:
    # if self.transform: ... transformed = self.transform(image=image, mask=mask)
    # This implies using `albumentations` style or custom dictionary based transforms.
    # To keep it simple and dependancy-free, let's just HARDCODE the resizing in dataset.py or 
    # Create a custom wrapper here.
    
    # Let's use a custom Collate Function or Wrapper Dataset to handle resizing.
    
    class ResizingWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, size=(256, 256)):
            self.dataset = dataset
            self.size = size
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            img, mask = self.dataset[idx] # These are tensors from the base dataset (loaded via cv2 -> tensor)
            
            # dataset.py returns Tensors already? 
            # Reviewing dataset.py:
            # "image = transforms.ToTensor()(image)" -> This makes it (C,H,W) float [0,1]
            # "mask = transforms.ToTensor()(mask)" -> This makes it (1,H,W) float [0,1]
            
            img = nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
            mask = nn.functional.interpolate(mask.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
            
            return img, mask
            
    train_dataset = ResizingWrapper(full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model
    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Save
    save_path = "segmentation_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Allow running directly
    train_segmentation()
