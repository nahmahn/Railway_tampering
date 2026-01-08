import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from tqdm import tqdm

from dataset import get_fault_dataset

def train_fault(num_epochs=10, batch_size=16, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'Railway Track fault Detection Updated', 'Train')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = get_fault_dataset(ROOT_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f"Starting Training for classes: {dataset.classes}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f} Acc: {100 * correct / total:.2f}%")
        
    save_path = "fault_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_fault()
